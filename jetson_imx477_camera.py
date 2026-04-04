#!/usr/bin/env python3
"""Simple Jetson IMX477 preview with optional autofocus."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

SYSTEM_NUMPY_PATH = "/usr/lib/python3/dist-packages"
SYSTEM_CV2_PATH = "/usr/lib/python3.10/dist-packages"
ARDUCAM_AF_LENS_PATH = "/home/nurman/MIPI_Camera/Jetson/IMX477/AF_LENS"

for package_path in (SYSTEM_CV2_PATH, SYSTEM_NUMPY_PATH, ARDUCAM_AF_LENS_PATH):
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

import cv2
import numpy

from Focuser import Focuser
from jetson_imx477_autofocus import HillClimbAutofocus, draw_overlay as draw_af_overlay


STOP_REQUESTED = False


def handle_sigint(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def gstreamer_pipeline(
    capture_width: int,
    capture_height: int,
    display_width: int,
    display_height: int,
    framerate: int,
    flip_method: int,
) -> str:
    return (
        "nvarguscamerasrc tnr-mode=0 ee-mode=0 ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        "format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson IMX477 preview.")
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1080)
    parser.add_argument("--display-width", type=int, default=1920)
    parser.add_argument("--display-height", type=int, default=1080)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=0)
    parser.add_argument("--autofocus", action="store_true")
    parser.add_argument("--i2c-bus", type=int, default=10)
    parser.add_argument("--focus", type=int, default=350)
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Record a 10 second MP4 clip into the caps folder.",
    )
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    if "GStreamer:                   YES" not in cv2.getBuildInformation():
        raise RuntimeError("This OpenCV build does not include GStreamer support.")

    pipeline = gstreamer_pipeline(
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        display_width=args.display_width,
        display_height=args.display_height,
        framerate=args.framerate,
        flip_method=args.flip_method,
    )
    print("Opening pipeline:")
    print(pipeline)

    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        raise RuntimeError("Failed to open the CSI camera.")
    return camera


def draw_preview_overlay(
    frame,
    autofocus_enabled: bool,
    focus_value: int | None,
    fps: float | None,
) -> None:
    lines = [
        f"AF: {'ON' if autofocus_enabled else 'OFF'}",
        f"Focus: {focus_value if focus_value is not None else 'n/a'}",
        f"FPS: {fps:.1f}" if fps is not None else "FPS: measuring...",
    ]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 30


def create_video_writer(args: argparse.Namespace):
    caps_dir = Path("/home/nurman/Desktop/archangel/caps")
    caps_dir.mkdir(parents=True, exist_ok=True)
    output_path = caps_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(args.framerate),
        (args.display_width, args.display_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    return writer, output_path


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"Python executable: {sys.executable}")
    print(f"NumPy: {numpy.__version__} ({numpy.__file__})")
    print(f"OpenCV: {cv2.__version__} ({cv2.__file__})")

    camera = open_camera(args)
    focuser = Focuser(args.i2c_bus) if args.autofocus else None
    autofocus = HillClimbAutofocus(focuser, args.focus) if focuser is not None else None
    writer = None
    output_path = None
    capture_started_at = None

    if args.capture:
        writer, output_path = create_video_writer(args)
        capture_started_at = time.monotonic()
        print(f"Recording 10 second clip to {output_path}")

    frame_count = 0
    last_fps = None
    fps_window_start = time.monotonic()

    try:
        while not STOP_REQUESTED:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed while preview was running.")

            frame_count += 1
            now = time.monotonic()
            elapsed = now - fps_window_start
            if elapsed >= 5.0:
                last_fps = frame_count / elapsed
                frame_count = 0
                fps_window_start = now

            if autofocus is not None:
                autofocus.update(frame)
                draw_af_overlay(frame, autofocus, last_fps)
            else:
                draw_preview_overlay(frame, False, None, last_fps)

            if writer is not None:
                writer.write(frame)
                if capture_started_at is not None and time.monotonic() - capture_started_at >= 10.0:
                    print(f"Saved clip to {output_path}")
                    break

            cv2.imshow("Jetson IMX477 Preview", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        if writer is not None:
            writer.release()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
