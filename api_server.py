#!/usr/bin/env python3
"""
ArchAngel API Server — bridges Jetson TCP stream to frontend WebSocket.

Runs the TRIBE v2 inference engine and exposes:
  - TCP server on port 5000 for Jetson frame streaming
  - HTTP/WebSocket on port 8000 for the Next.js frontend

Usage:
  python api_server.py
  python api_server.py --variant vitg --window-size 16 --stride 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Inference engine
from pc_inference_server import (
    StreamingBrainEngine,
    get_emotion_scores,
    get_top_regions,
    _get_visual_mask,
)
from stream_protocol import recv_message, send_message

# ── Shared state: latest prediction pushed to all WebSocket clients ──

_latest_prediction: dict = {}
_prediction_seq: int = 0
_latest_frame_jpg: bytes = b""
_ws_clients: set = set()
_prediction_lock = threading.Lock()


def _broadcast_to_ws(data: dict):
    """Store latest prediction for WebSocket clients."""
    global _latest_prediction, _prediction_seq
    with _prediction_lock:
        _prediction_seq += 1
        _latest_prediction = data


def _store_frame(jpg_bytes: bytes):
    """Store latest JPEG frame for the frontend."""
    global _latest_frame_jpg
    _latest_frame_jpg = jpg_bytes


# ── TCP server thread (handles Jetson connection) ──────────────────

def tcp_server_thread(engine: StreamingBrainEngine, args: argparse.Namespace):
    """Run the TCP server for Jetson in a background thread."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", args.tcp_port))
        server.listen(1)
        print(f"  TCP server listening on 0.0.0.0:{args.tcp_port} (Jetson)")

        while True:
            client, address = server.accept()
            print(f"\n  Jetson connected: {address[0]}:{address[1]}")
            with client:
                try:
                    handle_jetson_client(client, engine, args)
                except Exception as e:
                    print(f"\n  Jetson disconnected: {e}")


def handle_jetson_client(client: socket.socket, engine: StreamingBrainEngine,
                         args: argparse.Namespace):
    """Handle one Jetson client — sliding window + inference + broadcast to WS."""
    window: deque[np.ndarray] = deque(maxlen=args.window_size)
    engine.clip_history.clear()
    frames_since_inference = 0
    inference_count = 0
    total_frames = 0

    # Timeline buffer for the frontend chart
    timeline: deque[dict] = deque(maxlen=120)  # ~2 minutes of predictions

    while True:
        message, payload = recv_message(client)
        if message.get("type") != "frame":
            continue

        array = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        window.append(frame)
        frames_since_inference += 1
        total_frames += 1

        # Store frame for frontend video feed
        if total_frames % 3 == 0:  # every 3rd frame to save bandwidth
            _store_frame(payload)

        # Buffering phase
        if len(window) < args.window_size:
            send_message(client, {
                "type": "result", "status": "buffering",
                "window_fill": len(window), "window_size": args.window_size,
                "frame_id": message.get("frame_id", 0),
            })
            _broadcast_to_ws({"status": "buffering", "fill": len(window), "total": args.window_size})
            continue

        if frames_since_inference < args.stride:
            continue

        # ── Run inference ──
        frames_since_inference = 0
        inference_count += 1
        t0 = time.time()

        # Encode current window
        clip_feat = engine.encode_window(list(window))
        engine.clip_history.append(clip_feat)
        T = len(engine.clip_history)

        # Brain model on full history
        from einops import rearrange
        video_feat = torch.stack(list(engine.clip_history), dim=-1)
        audio_feat = torch.zeros(2, 1024, T)

        class B:
            pass
        batch = B()
        batch.data = {
            "video": video_feat.unsqueeze(0).to(engine.engine.brain_device),
            "audio": audio_feat.unsqueeze(0).to(engine.engine.brain_device),
            "subject_id": torch.zeros(1, dtype=torch.long, device=engine.engine.brain_device),
        }
        with torch.inference_mode():
            y_pred = engine.engine.brain_model(batch).detach().cpu().numpy()
        preds = rearrange(y_pred, "b d t -> (b t) d")

        now_n = min(3, len(preds))
        now_preds = preds[-now_n:]
        baseline_preds = preds[:-now_n] if len(preds) > 5 else preds

        # Emotion scores
        now_emo = get_emotion_scores(now_preds)
        base_emo = get_emotion_scores(baseline_preds)

        # Top regions (emotion only)
        top_regions = get_top_regions(now_preds, top_k=10, emotion_only=False)

        # System scores for frontend
        from pc_inference_server import _get_atlas, EMOTION_SYSTEMS
        cache = _get_atlas()
        mean_act = now_preds.mean(axis=0)

        # Build system scores matching frontend format
        SYSTEM_MAP = {
            "Visual Processing": ["G_cuneus", "G_occipital_middle", "G_occipital_sup", "Pole_occipital", "S_calcarine"],
            "Object/Face Recognition": ["G_oc-temp_lat-fusifor", "S_oc-temp_lat"],
            "Attention": ["G_parietal_sup", "S_intrapariet_and_P_trans"],
            "Social Cognition": ["S_temporal_sup", "Pole_temporal"],
            "Motor System": ["G_precentral", "G_and_S_paracentral"],
            "Higher Cognition": ["G_pariet_inf-Angular", "G_front_middle"],
            "Touch & Sensation": ["G_postcentral", "S_postcentral"],
            "Memory": ["G_oc-temp_med-Parahip", "S_collat_transv_ant"],
            "Emotion": ["G_insular_short", "G_Ins_lg_and_S_cent_ins", "G_and_S_cingul-Ant", "G_subcallosal"],
        }

        system_scores = []
        for sys_name, region_names in SYSTEM_MAP.items():
            vals = []
            for rn in region_names:
                if rn in cache["region_masks"]:
                    mask = cache["region_masks"][rn]
                    if mask.sum() > 0:
                        vals.append(float(np.abs(mean_act[mask]).mean()))
            score = np.mean(vals) if vals else 0.0
            system_scores.append({"system": sys_name, "score": round(score, 4)})

        # Spike detection
        weights = {"fear_anxiety": 3.0, "anger_stress": 2.0, "emotional_arousal": 2.0, "social_emotion": 1.0}
        now_c = sum(now_emo.get(k, 0) * w for k, w in weights.items() if k in now_emo) / sum(weights.values())
        base_c = sum(base_emo.get(k, 0) * w for k, w in weights.items() if k in base_emo) / sum(weights.values())

        now_vis = now_emo.get("visual", 0.01)
        base_vis = base_emo.get("visual", 0.01)
        vis_chg = ((now_vis - base_vis) / max(base_vis, 0.001)) * 100
        emo_chg = ((now_c - base_c) / max(base_c, 0.001)) * 100
        net_pct = emo_chg - vis_chg * 0.5 if abs(vis_chg) > 5 else emo_chg

        if abs(net_pct) > 25:
            spike = "SPIKE!" if net_pct > 0 else "DROP!"
            alert_level = "ALERT"
        elif abs(net_pct) > 12:
            spike = "rising" if net_pct > 0 else "falling"
            alert_level = "WATCH"
        else:
            spike = "steady"
            alert_level = "NORMAL"

        dt = time.time() - t0

        # Timeline point
        t_secs = inference_count * args.stride / 30.0  # approximate seconds
        tp = {"time": round(t_secs, 1), "total": round(float(np.abs(mean_act).mean()), 4)}
        tp["systems"] = {s["system"]: s["score"] for s in system_scores}
        timeline.append(tp)

        # Dominant system
        dominant = max(system_scores, key=lambda s: s["score"])

        # Alert label
        fear_val = now_emo.get("fear_anxiety", 0)
        anger_val = now_emo.get("anger_stress", 0)
        arousal_val = now_emo.get("emotional_arousal", 0)
        if alert_level == "ALERT":
            alert_label = "Emotional spike detected"
        elif alert_level == "WATCH":
            alert_label = "Elevated emotional activity"
        else:
            alert_label = "Baseline activity"

        # Build prediction matching BrainPrediction TypeScript interface
        prediction = {
            "status": "ok",
            "inference_count": inference_count,
            "history_length": T,
            "systemScores": system_scores,
            "alertLevel": alert_level,
            "alertLabel": alert_label,
            "topRegions": [
                {"rank": i + 1, "name": r["region"], "label": r.get("category", ""), "score": r["score"]}
                for i, r in enumerate(top_regions[:10])
            ],
            "totalActivity": round(float(np.abs(mean_act).mean()), 4),
            "dominantSystem": dominant["system"],
            "engagement": f"{now_c:.4f}",
            "timeline": list(timeline),
            "spike": spike,
            "spike_pct": round(net_pct, 1),
            "emotions": {k: round(v, 4) for k, v in now_emo.items()},
            "timing_ms": round(dt * 1000),
        }

        # Send back to Jetson
        send_message(client, {
            "type": "result", "status": "ok",
            "frame_id": message.get("frame_id", 0),
            "inference_count": inference_count,
            "emotion_score": round(now_c, 4),
            "spike": spike, "spike_pct": round(net_pct, 1),
            "emotions": {k: round(v, 4) for k, v in now_emo.items()},
            "top_regions": [{"region": r["region"], "score": r["score"], "category": r.get("category", "")} for r in top_regions[:5]],
            "timing": {"total_ms": round(dt * 1000)},
        })

        # Broadcast to WebSocket clients
        _broadcast_to_ws(prediction)

        # Log
        print(f"  #{inference_count:4d} | {dt*1000:.0f}ms | T={T:2d} | "
              f"fear={fear_val:.4f} anger={anger_val:.4f} arousal={arousal_val:.4f} "
              f"[{spike:7s} {net_pct:+5.1f}%] | {dominant['system']}")


# ── Starlette WebSocket + HTTP server (no middleware = no 403) ─────

def create_app():
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute, Route
    from starlette.responses import JSONResponse, Response
    from starlette.websockets import WebSocket, WebSocketDisconnect

    async def ws_handler(ws: WebSocket):
        await ws.accept()
        _ws_clients.add(ws)
        print(f"  Frontend connected via WebSocket ({len(_ws_clients)} clients)")
        last_seq = 0
        try:
            while True:
                with _prediction_lock:
                    seq = _prediction_seq
                    data = _latest_prediction.copy() if _latest_prediction else None
                if data and seq != last_seq:
                    last_seq = seq
                    await ws.send_json(data)
                await asyncio.sleep(0.15)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"  WS error: {e}")
        finally:
            _ws_clients.discard(ws)
            print(f"  Frontend disconnected ({len(_ws_clients)} clients)")

    async def get_frame(request):
        headers = {"Access-Control-Allow-Origin": "*"}
        if not _latest_frame_jpg:
            return Response(content=b"", media_type="image/jpeg", status_code=204, headers=headers)
        return Response(content=_latest_frame_jpg, media_type="image/jpeg", headers=headers)

    async def get_status(request):
        with _prediction_lock:
            data = _latest_prediction or {"status": "waiting_for_jetson"}
        return JSONResponse(data, headers={"Access-Control-Allow-Origin": "*"})

    return Starlette(routes=[
        WebSocketRoute("/ws", ws_handler),
        Route("/api/frame", get_frame),
        Route("/api/status", get_status),
    ])


# ── Main ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ArchAngel API Server")
    p.add_argument("--tcp-port", type=int, default=5000)
    p.add_argument("--http-port", type=int, default=8000)
    p.add_argument("--window-size", type=int, default=16)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--variant", default="vitg", choices=["vitg", "vitl", "vith"])
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Load models
    engine = StreamingBrainEngine(variant=args.variant, compile_video=args.compile)

    # Start TCP server thread
    tcp_thread = threading.Thread(target=tcp_server_thread, args=(engine, args), daemon=True)
    tcp_thread.start()

    # Start FastAPI server
    import uvicorn
    app = create_app()
    print(f"  HTTP/WebSocket server on http://0.0.0.0:{args.http_port}")
    print(f"  Frontend WebSocket: ws://0.0.0.0:{args.http_port}/ws")
    print(f"\n  Ready. Connect Jetson to TCP:{args.tcp_port}, Frontend to WS:{args.http_port}/ws\n")
    uvicorn.run(app, host="0.0.0.0", port=args.http_port, log_level="warning")


if __name__ == "__main__":
    main()
