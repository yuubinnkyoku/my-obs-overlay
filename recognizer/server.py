import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional, Set

import sounddevice as sd
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.yaml"
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="OBS BGM Recognizer", version="0.1.0")


class AppConfig(BaseModel):
    device_name: Optional[str] = Field(default=None, description="Input device name (WASAPI)")
    bgm_dir: Optional[str] = Field(default=None, description="Directory containing reference BGM files")


# ------------------ Config Load/Save ------------------

def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return AppConfig(**data)
        except Exception:
            return AppConfig()
    return AppConfig()


def save_config(cfg: AppConfig) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(exclude_none=True), f, allow_unicode=True, sort_keys=False)


# ------------------ Devices API ------------------

@app.get("/api/devices")
def list_input_devices():
    devices = sd.query_devices()
    inputs = []
    for dev in devices:
        try:
            if dev.get("max_input_channels", 0) > 0:
                # Windows WASAPI names may be long; include index for clarity
                inputs.append({
                    "name": dev.get("name"),
                    "hostapi": dev.get("hostapi"),
                    "max_input_channels": dev.get("max_input_channels", 0),
                    "default_samplerate": dev.get("default_samplerate", None),
                })
        except Exception:
            continue
    return {"devices": inputs}


@app.get("/api/config")
def get_config():
    cfg = load_config()
    return cfg.model_dump()


class SaveConfigBody(BaseModel):
    device_name: Optional[str] = None
    bgm_dir: Optional[str] = None


@app.post("/api/save-config")
def save_config_endpoint(body: SaveConfigBody):
    cfg = load_config()
    if body.device_name is not None:
        cfg.device_name = body.device_name
    if body.bgm_dir is not None:
        cfg.bgm_dir = body.bgm_dir
    save_config(cfg)
    return {"ok": True}


# ------------------ Static Config Page ------------------

@app.get("/config")
def config_page():
    html = STATIC_DIR / "config.html"
    if not html.exists():
        return HTMLResponse("<h1>Config page not found</h1>", status_code=404)
    return FileResponse(str(html))


# ------------------ WebSocket (track broadcast) ------------------

class WSHub:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict):
        dead = []
        payload = json.dumps(message, ensure_ascii=False)
        async with self._lock:
            for ws in list(self._clients):
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)


ws_hub = WSHub()


@app.websocket("/")
async def ws_root(ws: WebSocket):
    await ws_hub.add(ws)
    try:
        while True:
            # We do not expect client messages; keep connection alive.
            await ws.receive_text()
    except WebSocketDisconnect:
        await ws_hub.remove(ws)
    except Exception:
        await ws_hub.remove(ws)


# Test endpoint to push a demo track to clients (useful for verifying end-to-end)
class PushBody(BaseModel):
    title: str
    series: Optional[str] = ""
    composer: Optional[str] = ""
    confidence: Optional[float] = 0.0


@app.post("/api/push")
async def push_track(body: PushBody):
    await ws_hub.broadcast(body.model_dump())
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
