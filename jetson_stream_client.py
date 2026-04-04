#!/usr/bin/env python3
"""Capture frames on Jetson, stream them to a PC, and receive results back."""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace

# Ensure system OpenCV (with GStreamer) is found before any pip-installed version
for _p in ("/usr/lib/python3.10/dist-packages", "/usr/lib/python3/dist-packages"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2

from jetson_imx477_camera import draw_preview_overlay, open_camera
from stream_protocol import recv_message, send_message


STOP_REQUESTED = False


def handle_sigint(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Jetson camera frames to a PC.")
    parser.add_argument("--host", required=True, help="PC IP address or hostname.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--send-every", type=int, default=1, help="Send every Nth frame.")
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1080)
    parser.add_argument("--display-width", type=int, default=1920)
    parser.add_argument("--display-height", type=int, default=1080)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=0)
    return parser.parse_args()


def make_camera_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        display_width=args.display_width,
        display_height=args.display_height,
        framerate=args.framerate,
        flip_method=args.flip_method,
        autofocus=False,
        i2c_bus=10,
        focus=350,
        capture=False,
    )


def result_receiver(sock: socket.socket, latest_result: dict) -> None:
    while not STOP_REQUESTED:
        try:
            message, _ = recv_message(sock)
        except ConnectionError:
            return
        if message.get("type") == "result":
            # Only overwrite with actual predictions, not buffering/waiting acks
            if message.get("status") == "ok":
                latest_result.clear()
                latest_result.update(message)
            elif not latest_result:
                # Show buffering status if we haven't gotten a prediction yet
                latest_result.clear()
                latest_result.update(message)


def draw_result_overlay(frame, latest_result: dict) -> None:
    if not latest_result:
        return

    status = latest_result.get("status", "?")

    if status == "buffering":
        fill = latest_result.get("window_fill", 0)
        size = latest_result.get("window_size", 30)
        lines = [f"Buffering: {fill}/{size} frames"]
        color = (0, 165, 255)  # orange
    elif status == "waiting":
        left = latest_result.get("frames_until_next", "?")
        lines = [f"Next prediction in {left} frames"]
        color = (200, 200, 200)  # gray
    elif status == "ok":
        timing = latest_result.get("timing", {})
        emo_score = latest_result.get("emotion_score", 0)
        spike = latest_result.get("spike", "steady")
        spike_pct = latest_result.get("spike_pct", 0)
        emotions = latest_result.get("emotions", {})
        regions = latest_result.get("top_regions", [])
        inference_n = latest_result.get("inference_count", 0)

        fear = emotions.get("fear_anxiety", 0)
        anger = emotions.get("anger_stress", 0)
        arousal = emotions.get("emotional_arousal", 0)
        social = emotions.get("social_emotion", 0)

        # Spike indicator
        if spike in ("SPIKE!", "DROP!"):
            spike_text = f"  !! {spike} ({spike_pct:+.0f}%) !!"
        elif spike in ("rising", "falling"):
            spike_text = f"  ({spike} {spike_pct:+.0f}%)"
        else:
            spike_text = ""

        lines = [
            f"#{inference_n} | {timing.get('total_ms', 0):.0f}ms | {spike}{spike_text}",
            f"Fear:{fear:.3f} Anger:{anger:.3f} Arousal:{arousal:.3f}",
        ]
        for i, r in enumerate(regions[:2]):
            name = r.get("region", "?")
            score = r.get("score", 0)
            lines.append(f"  {name}: {score:+.3f}")

        if spike == "SPIKE!":
            color = (0, 0, 255)  # red
        elif spike == "DROP!":
            color = (255, 0, 0)  # blue
        elif spike in ("rising",):
            color = (0, 140, 255)  # orange
        else:
            color = (0, 255, 0)  # green
    else:
        lines = [f"Status: {status}"]
        color = (255, 255, 255)

    y = 140
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 24


def main() -> int:
    global STOP_REQUESTED
    args = parse_args()
    signal.signal(signal.SIGINT, handle_sigint)

    camera = open_camera(make_camera_args(args))
    sock = socket.create_connection((args.host, args.port))
    latest_result: dict = {}
    receiver = threading.Thread(target=result_receiver, args=(sock, latest_result), daemon=True)
    receiver.start()

    frame_id = 0
    preview_fps = None
    fps_window_start = time.monotonic()
    frames_in_window = 0

    try:
        while not STOP_REQUESTED:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed while streaming.")

            frame_id += 1
            frames_in_window += 1
            elapsed = time.monotonic() - fps_window_start
            if elapsed >= 5.0:
                preview_fps = frames_in_window / elapsed
                frames_in_window = 0
                fps_window_start = time.monotonic()

            if frame_id % args.send_every == 0:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed.")

                send_message(
                    sock,
                    {
                        "type": "frame",
                        "frame_id": frame_id,
                        "timestamp_ms": int(time.time() * 1000),
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                    },
                    encoded.tobytes(),
                )

            if args.show_preview:
                draw_preview_overlay(frame, False, None, preview_fps)
                draw_result_overlay(frame, latest_result)
                cv2.imshow("Jetson Stream Client", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        STOP_REQUESTED = True
        sock.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
