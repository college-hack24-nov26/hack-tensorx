"""
routes/simulator_stream.py

Bridges the SIMULATOR (port 8002) into ECGX:
  - POST /live/start  -> connects to wss://<simulator_host>/ws/<session_id> (ssl bypass for self-signed)
  - POST /live/stop   -> cancels bridge
  - GET  /live/status -> bridge state
  - WS   /ws/live     -> ECGX dashboard clients subscribe here for live data + predictions
"""
import asyncio
import json
import ssl
import time
from collections import deque
from typing import Optional

import numpy as np
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from services.model_loader import predict_signal
from services.preprocess import preprocess_signal, clean_ecg_signal
from services.webhook_service import async_trigger_buzzer, async_trigger_emergency_webhook, CRITICAL_CONDITIONS

router = APIRouter()

_dashboard_clients: set[WebSocket] = set()
_bridge_task: Optional[asyncio.Task] = None
_bridge_session: Optional[str] = None
_bridge_host: Optional[str] = None

WINDOW = 2500
PREDICT_EVERY = 250  # predict every 250 new samples = every 1 second at 250Hz
ALERT_COOLDOWN = 30  # seconds between repeated alerts for same condition

# SSL context that bypasses self-signed cert verification (simulator uses self-signed)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

import aiohttp

async def _get_active_session(simulator_host: str) -> Optional[str]:
    """Fetch the active session ID from the simulator REST API."""
    for scheme in ("http", "https"):
        try:
            url = f"{scheme}://{simulator_host}/api/session/active"
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("session_id")
        except Exception:
            continue
    return None


async def _broadcast(msg: dict):
    dead = set()
    data = json.dumps(msg)
    for ws in _dashboard_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    _dashboard_clients.difference_update(dead)


async def _bridge_loop(simulator_host: str, session_id: str):
    global _bridge_session, _bridge_host
    rolling: deque = deque(maxlen=WINDOW)
    new_since_pred = 0
    last_alert_time: dict = {}  # prediction -> last trigger timestamp

    # Validate session ID against simulator's active session
    active_id = await _get_active_session(simulator_host)
    if active_id is None:
        await _broadcast({"type": "bridge_status", "status": "error", "message": "Cannot reach simulator REST API"})
        _bridge_session = None
        _bridge_host = None
        return
    if active_id != session_id:
        await _broadcast({
            "type": "bridge_status",
            "status": "error",
            "message": f"Invalid session ID. Active session is: {active_id}"
        })
        _bridge_session = None
        _bridge_host = None
        return

    # Try wss:// first (simulator uses self-signed SSL), fall back to ws://
    for scheme, kwargs in [("wss", {"ssl": _ssl_ctx}), ("ws", {})]:
        uri = f"{scheme}://{simulator_host}/ws/{session_id}"
        await _broadcast({"type": "bridge_status", "status": "connecting", "uri": uri})
        try:
            async with websockets.connect(uri, ping_interval=20, **kwargs) as ws_sim:
                await _broadcast({"type": "bridge_status", "status": "connected", "uri": uri})
                async for raw in ws_sim:
                    msg = json.loads(raw)

                    if msg.get("type") == "stopped":
                        await _broadcast({"type": "bridge_status", "status": "stopped"})
                        return

                    if msg.get("type") == "segment":
                        await _broadcast({
                            "type": "segment",
                            "rhythm": msg.get("rhythm", ""),
                            "source": msg.get("source", ""),
                            "segment_idx": msg.get("segment_idx", 0),
                        })
                        continue

                    if msg.get("type") != "samples":
                        continue

                    samples: list = msg["samples"]
                    for s in samples:
                        rolling.append(float(s))
                    new_since_pred += len(samples)

                    prediction_update = None
                    if len(rolling) == WINDOW and new_since_pred >= PREDICT_EVERY:
                        new_since_pred = 0
                        arr = np.array(rolling, dtype=np.float32)
                        try:
                            result = predict_signal(preprocess_signal(arr))
                            prediction_update = {
                                "prediction": result["prediction"],
                                "probabilities": result["probabilities"],
                                "filtered_signal": clean_ecg_signal(arr).tolist(),
                            }
                            # ── Alert triggers (ESP32 + n8n) ──────────────────
                            pred = result["prediction"]
                            if pred in CRITICAL_CONDITIONS:
                                now = time.time()
                                last = last_alert_time.get(pred, 0)
                                if now - last >= ALERT_COOLDOWN:
                                    last_alert_time[pred] = now
                                    rhythm = msg.get("rhythm", "")
                                    source = msg.get("source", "")
                                    # Fire both triggers concurrently, don't block stream
                                    asyncio.create_task(async_trigger_buzzer(on=True))
                                    asyncio.create_task(async_trigger_emergency_webhook(
                                        pred, result["probabilities"], rhythm, source, session_id
                                    ))
                                    # Broadcast alert event to all dashboard clients
                                    await _broadcast({
                                        "type": "alert",
                                        "prediction": pred,
                                        "confidence": round(result["probabilities"].get(pred, 0) * 100, 1),
                                        "rhythm": rhythm,
                                        "source": source,
                                        "esp_triggered": True,
                                        "webhook_triggered": True,
                                    })
                        except Exception as e:
                            prediction_update = {"error": str(e)}

                    out = {
                        "type": "live",
                        "samples": samples,
                        "samples_sent": msg.get("samples_sent", 0),
                        "elapsed": msg.get("elapsed", 0),
                        "rhythm": msg.get("rhythm", ""),
                        "source": msg.get("source", ""),
                        "session_id": session_id,
                    }
                    if prediction_update:
                        out["prediction_update"] = prediction_update

                    await _broadcast(out)
            # Connected and finished cleanly — don't try fallback
            return
        except Exception as e:
            if scheme == "wss":
                # wss failed, try ws fallback
                print(f"[bridge] wss failed ({e}), trying ws://")
                continue
            await _broadcast({"type": "bridge_status", "status": "error", "message": str(e)})
    _bridge_session = None
    _bridge_host = None


class BridgeRequest(BaseModel):
    simulator_host: str = "localhost:8002"
    session_id: str


@router.post("/live/start")
async def start_bridge(req: BridgeRequest):
    global _bridge_task, _bridge_session, _bridge_host
    if _bridge_task and not _bridge_task.done():
        _bridge_task.cancel()
    _bridge_session = req.session_id
    _bridge_host = req.simulator_host
    _bridge_task = asyncio.create_task(_bridge_loop(req.simulator_host, req.session_id))
    return {"status": "bridge_started", "session_id": req.session_id, "host": req.simulator_host}


@router.post("/live/stop")
async def stop_bridge():
    global _bridge_task
    if _bridge_task and not _bridge_task.done():
        _bridge_task.cancel()
    await _broadcast({"type": "bridge_status", "status": "stopped"})
    return {"status": "stopped"}


@router.get("/live/active-session")
async def get_active_session(simulator_host: str = "10.246.13.184:8002"):
    """Proxy the simulator's active session to the dashboard."""
    session_id = await _get_active_session(simulator_host)
    if session_id:
        return {"session_id": session_id}
    return {"session_id": None}


@router.get("/live/status")
def bridge_status():
    return {
        "running": bool(_bridge_task and not _bridge_task.done()),
        "session_id": _bridge_session,
        "simulator_host": _bridge_host,
        "dashboard_clients": len(_dashboard_clients),
    }


@router.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    _dashboard_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _dashboard_clients.discard(websocket)
