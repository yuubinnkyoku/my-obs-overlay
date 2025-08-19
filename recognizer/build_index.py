import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import librosa
import faiss  # type: ignore

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.yaml"
INDEX_DIR = APP_DIR / "index"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def parse_metadata(bgm_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Optional: read metadata.csv if provided (file,title,series,composer)
    Return mapping by absolute file path string.
    """
    csv_path = bgm_dir / "metadata.csv"
    meta: Dict[str, Dict[str, str]] = {}
    if not csv_path.exists():
        return meta
    try:
        import csv

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file = row.get("file") or row.get("path")
                if not file:
                    continue
                abs_path = str((bgm_dir / file).resolve())
                meta[abs_path] = {
                    "title": row.get("title", ""),
                    "series": row.get("series", ""),
                    "composer": row.get("composer", ""),
                }
    except Exception:
        pass
    return meta


def feature_vector(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute a compact feature vector for matching (cosine similarity).
    - Chroma CENS (12-d) mean
    - MFCC (13-d) mean
    - Spectral centroid mean (1)
    - Rolloff (0.85) mean (1)
    Total: 27 dims
    """
    if y.size == 0:
        return np.zeros(27, dtype=np.float32)
    # Pre-emphasis light
    y = librosa.effects.preemphasis(y)
    # Chroma
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    # Spectral features
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    vec = np.concatenate([
        chroma_mean,
        mfcc_mean,
        np.array([np.mean(cent), np.mean(roll)]),
    ]).astype(np.float32)
    # Normalize (z-score per vector)
    v = vec
    mu = float(v.mean())
    sigma = float(v.std() + 1e-8)
    v = (v - mu) / sigma
    # L2 normalize for cosine via inner product
    nrm = float(np.linalg.norm(v) + 1e-8)
    return (v / nrm).astype(np.float32)


def infer_metadata(path: Path) -> Dict[str, str]:
    title = path.stem
    series = path.parent.name
    return {"title": title, "series": series, "composer": ""}


def main():
    cfg = load_config()
    bgm_dir = cfg.get("bgm_dir")
    sr = int(cfg.get("sample_rate", 48000))
    max_dur = float(cfg.get("index_max_duration_sec", 60.0))  # analyze up to N seconds per file
    index_path_cfg = cfg.get("index_path", INDEX_DIR / "faiss.index")
    metadata_path_cfg = cfg.get("metadata_path", INDEX_DIR / "metadata.json")
    index_path = Path(index_path_cfg)
    metadata_path = Path(metadata_path_cfg)

    if not bgm_dir:
        raise SystemExit("config.yaml: bgm_dir is not set. Open /config and save a directory.")

    # Resolve paths relative to recognizer/ (APP_DIR)
    bgm_dir_str = str(bgm_dir)
    root = Path(bgm_dir_str)
    if not root.is_absolute():
        root = (APP_DIR / root).resolve()
    if not index_path.is_absolute():
        index_path = (APP_DIR / index_path).resolve()
    if not metadata_path.is_absolute():
        metadata_path = (APP_DIR / metadata_path).resolve()
    if not root.exists():
        raise SystemExit(f"BGM directory not found: {root}")

    files = discover_audio_files(root)
    if not files:
        raise SystemExit("No audio files found in bgm_dir.")

    print(f"Found {len(files)} audio files. Building features (<= {max_dur:.0f}s per file)...")

    feats: List[np.ndarray] = []
    metas: List[Dict[str, str]] = []

    csv_meta = parse_metadata(root)

    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            # Limit duration to speed up indexing on long tracks
            y, _ = librosa.load(str(f), sr=sr, mono=True, duration=max_dur)
            v = feature_vector(y, sr)
            feats.append(v)
            abs_path = str(f.resolve())
            m = csv_meta.get(abs_path) or infer_metadata(f)
            m = {
                "path": abs_path,
                "title": m.get("title", f.stem),
                "series": m.get("series", f.parent.name),
                "composer": m.get("composer", ""),
            }
            metas.append(m)
        except Exception as e:
            print(f"[WARN] failed on {f}: {e}")
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  progress: {i}/{len(files)} files processed in {elapsed:.1f}s")

    X = np.vstack(feats).astype(np.float32)

    # Cosine via inner product on L2-normalized vectors
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"Saved index -> {index_path}")
    print(f"Saved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
