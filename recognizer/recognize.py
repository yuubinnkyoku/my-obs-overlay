import json
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import faiss  # type: ignore
import httpx
import librosa
import numpy as np
import sounddevice as sd
import yaml

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def feature_vector(y: np.ndarray, sr: int) -> np.ndarray:
    """Mirror of build_index.py feature extractor (27 dims, L2-normalized)."""
    if y.size == 0:
        return np.zeros(27, dtype=np.float32)
    y = librosa.effects.preemphasis(y)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    vec = np.concatenate([
        chroma_mean,
        mfcc_mean,
        np.array([np.mean(cent), np.mean(roll)]),
    ]).astype(np.float32)
    mu = float(vec.mean())
    sigma = float(vec.std() + 1e-8)
    v = (vec - mu) / sigma
    nrm = float(np.linalg.norm(v) + 1e-8)
    return (v / nrm).astype(np.float32)


def find_input_device(name_hint: Optional[str]) -> Optional[int]:
    devices = sd.query_devices()
    if name_hint:
        # best-effort substring match (case-insensitive)
        lowered = name_hint.lower()
        for i, d in enumerate(devices):
            try:
                if d.get("max_input_channels", 0) > 0 and lowered in str(d.get("name", "")).lower():
                    return i
            except Exception:
                continue
    # fallback: default input device index
    try:
        default_idx = sd.default.device[0]  # (input, output)
        if isinstance(default_idx, int):
            return default_idx
    except Exception:
        pass
    # last resort: first input-capable device
    for i, d in enumerate(devices):
        try:
            if d.get("max_input_channels", 0) > 0:
                return i
        except Exception:
            continue
    return None


def find_wasapi_loopback_device(name_hint: Optional[str]) -> Optional[int]:
    """Find a WASAPI output device index by name to use with loopback capture.

    The provided name_hint should be the base device name (without the
    ' (loopback)' suffix).
    """
    devices = sd.query_devices()
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []

    def hostapi_name(idx: int) -> str:
        try:
            return hostapis[idx].get("name", str(idx))
        except Exception:
            return str(idx)

    if name_hint:
        lowered = name_hint.lower()
        for i, d in enumerate(devices):
            try:
                if (
                    d.get("max_output_channels", 0) > 0
                    and hostapi_name(int(d.get("hostapi", 0))) == "Windows WASAPI"
                    and lowered in str(d.get("name", "")).lower()
                ):
                    return i
            except Exception:
                continue
    # Fallback: pick system default output (for WASAPI) if available
    try:
        default_out = sd.default.device[1]  # (input, output)
        if isinstance(default_out, int):
            d = devices[default_out]
            if hostapi_name(int(d.get("hostapi", 0))) == "Windows WASAPI":
                return default_out
    except Exception:
        pass
    # Last resort: any WASAPI output device
    for i, d in enumerate(devices):
        try:
            if d.get("max_output_channels", 0) > 0 and hostapi_name(int(d.get("hostapi", 0))) == "Windows WASAPI":
                return i
        except Exception:
            continue
    return None


class RingAudio:
    def __init__(self, sr: int, seconds: float, channels: int = 1) -> None:
        self.sr = sr
        self.channels = channels
        self.N = int(sr * seconds)
        self.buf = np.zeros((self.N, channels), dtype=np.float32)
        self.ptr = 0
        self.filled = False

    def write(self, frames: np.ndarray) -> None:
        # frames: shape (n, channels)
        n = frames.shape[0]
        if n >= self.N:
            self.buf[:, :] = frames[-self.N :, :]
            self.ptr = 0
            self.filled = True
            return
        end = self.ptr + n
        if end <= self.N:
            self.buf[self.ptr:end, :] = frames
        else:
            first = self.N - self.ptr
            self.buf[self.ptr:, :] = frames[:first, :]
            self.buf[: n - first, :] = frames[first:, :]
        self.ptr = (self.ptr + n) % self.N
        if self.ptr == 0:
            self.filled = True

    def read_mono(self) -> np.ndarray:
        if not self.filled and self.ptr == 0:
            return np.array([], dtype=np.float32)
        if not self.filled:
            data = self.buf[: self.ptr, :]
        else:
            data = np.vstack([self.buf[self.ptr :, :], self.buf[: self.ptr, :]])
        if self.channels > 1:
            mono = np.mean(data, axis=1)
        else:
            mono = data[:, 0]
        return mono.astype(np.float32)


def main() -> None:
    cfg = load_config()
    device_name = cfg.get("device_name")
    bgm_dir_cfg = cfg.get("bgm_dir")
    sr = int(cfg.get("sample_rate", 48000))
    channels = int(cfg.get("channels", 1))
    window_sec = float(cfg.get("window_sec", 3.0))
    hop_sec = float(cfg.get("hop_sec", 1.0))
    min_conf = float(cfg.get("min_confidence", 0.6))
    cooldown = float(cfg.get("announce_cooldown_sec", 3.0))

    index_path_cfg = cfg.get("index_path", APP_DIR / "index/faiss.index")
    meta_path_cfg = cfg.get("metadata_path", APP_DIR / "index/metadata.json")
    index_path = Path(index_path_cfg)
    meta_path = Path(meta_path_cfg)

    # Resolve relative paths against recognizer/ (APP_DIR)
    if bgm_dir_cfg and not Path(str(bgm_dir_cfg)).is_absolute():
        bgm_dir = (APP_DIR / str(bgm_dir_cfg)).resolve()
    else:
        bgm_dir = Path(str(bgm_dir_cfg)) if bgm_dir_cfg else None

    if not index_path.is_absolute():
        index_path = (APP_DIR / index_path).resolve()
    if not meta_path.is_absolute():
        meta_path = (APP_DIR / meta_path).resolve()

    if not index_path.exists() or not meta_path.exists():
        raise SystemExit("Index or metadata not found. Run build_index.py first.")

    print("Loading FAISS index and metadata...")
    index = faiss.read_index(str(index_path))
    metas = json.loads(meta_path.read_text(encoding="utf-8"))

    # Support normal inputs and WASAPI loopback pseudo-devices
    loopback = False
    device_idx: Optional[int]
    base_name = device_name
    if device_name and device_name.endswith(" (loopback)"):
        base_name = device_name[:-10]
        device_idx = find_wasapi_loopback_device(base_name)
        loopback = True
    else:
        device_idx = find_input_device(device_name)
    if device_idx is None:
        raise SystemExit("No suitable device found. Set device_name via /config.")

    ring = RingAudio(sr=sr, seconds=window_sec, channels=channels)

    # Simple gate to avoid processing silence
    def rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    last_announce: Tuple[Optional[int], float] = (None, 0.0)

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            # dropouts or overflows; still try to proceed
            pass
        arr = np.asarray(indata, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        ring.write(arr)

    print(f"Opening {'loopback' if loopback else 'input'} stream on device index {device_idx} (sr={sr}, ch={channels})...")
    if loopback:
        # Validate host API is WASAPI for the chosen device
        devices = sd.query_devices()
        try:
            hostapis = sd.query_hostapis()
        except Exception:
            hostapis = []

        def hostapi_name(idx: int) -> str:
            try:
                return hostapis[idx].get("name", str(idx))
            except Exception:
                return str(idx)

        d = devices[device_idx]
        api = hostapi_name(int(d.get("hostapi", 0)))
        if api != "Windows WASAPI":
            raise SystemExit(f"Selected device host API is '{api}', but WASAPI is required for loopback. Pick a '(loopback)' WASAPI output device.")

        # Check WasapiSettings availability
        WasapiSettings = getattr(sd, "WasapiSettings", None)
        if WasapiSettings is None:
            raise SystemExit(
                "sounddevice.WasapiSettings is not available. Ensure you're on Windows and have sounddevice >= 0.4.6 installed."
            )

        try:
            extra = WasapiSettings()
            if hasattr(extra, "loopback"):
                setattr(extra, "loopback", True)
            else:
                raise SystemExit("This sounddevice build does not expose 'loopback' on WasapiSettings.")
            stream_ctx = sd.InputStream(
                device=device_idx,
                samplerate=sr,
                channels=channels,
                dtype="float32",
                callback=callback,
                blocksize=0,
                extra_settings=extra,
            )
        except Exception as e:
            raise SystemExit(f"Failed to open WASAPI loopback stream: {e}")
    else:
        stream_ctx = sd.InputStream(
            device=device_idx,
            samplerate=sr,
            channels=channels,
            dtype="float32",
            callback=callback,
            blocksize=0,
        )

    with stream_ctx:
        next_t = time.time()
        while True:
            now = time.time()
            if now < next_t:
                time.sleep(0.01)
                continue
            next_t = now + hop_sec

            y = ring.read_mono()
            if y.size == 0:
                continue
            # Avoid short buffers that trigger STFT n_fft warnings and unstable features
            # Ensure we have at least 2048 samples (~43ms at 48kHz)
            if y.size < 2048:
                continue
            if rms(y) < 1e-3:
                # too quiet, skip to save CPU
                continue

            vec = feature_vector(y, sr)
            q = vec.reshape(1, -1).astype(np.float32)
            D, I = index.search(q, k=1)
            score = float(D[0][0])  # cosine in [-1, 1]
            idx = int(I[0][0])
            if idx < 0 or idx >= len(metas):
                continue

            # map cosine [-1,1] -> [0,1] if you prefer; here we use raw cosine threshold
            confidence = score
            if confidence < min_conf:
                continue

            last_idx, last_time = last_announce
            if last_idx == idx and (now - last_time) < cooldown:
                continue

            m = metas[idx]
            payload = {
                "title": m.get("title", "Unknown"),
                "series": m.get("series", ""),
                "composer": m.get("composer", ""),
                "confidence": round(confidence, 3),
            }
            try:
                # POST to server to broadcast via WS
                httpx.post("http://127.0.0.1:8765/api/push", json=payload, timeout=5.0)
                print(f"Announce: {payload}")
                last_announce = (idx, now)
            except Exception as e:
                print(f"[WARN] push failed: {e}")


if __name__ == "__main__":
    main()
