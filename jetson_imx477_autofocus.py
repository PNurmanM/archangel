#!/usr/bin/env python3
"""Simple autofocus helper for the Jetson IMX477 motorized lens."""

from __future__ import annotations

from collections import deque
import sys

SYSTEM_NUMPY_PATH = "/usr/lib/python3/dist-packages"
SYSTEM_CV2_PATH = "/usr/lib/python3.10/dist-packages"
ARDUCAM_AF_LENS_PATH = "/home/nurman/MIPI_Camera/Jetson/IMX477/AF_LENS"

for package_path in (SYSTEM_CV2_PATH, SYSTEM_NUMPY_PATH, ARDUCAM_AF_LENS_PATH):
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

import cv2
import numpy


FOCUS_MIN = 0
FOCUS_MAX = 1000
TRACK_STEP = 8
SEARCH_STEP = 28
MIN_STEP = 4
SETTLE_FRAMES = 3
CHECK_INTERVAL = 2
ROI_SCALE = 0.30
IMPROVE_RATIO = 0.02
SCENE_DROP_RATIO = 0.60
SCENE_DROP_COUNT = 3


def clamp_focus(value: int) -> int:
    return max(FOCUS_MIN, min(FOCUS_MAX, value))


def center_roi(frame):
    height, width = frame.shape[:2]
    roi_w = max(40, int(width * ROI_SCALE))
    roi_h = max(40, int(height * ROI_SCALE))
    x1 = (width - roi_w) // 2
    y1 = (height - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def sharpness_score(frame) -> tuple[float, tuple[int, int, int, int]]:
    roi, box = center_roi(frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = grad_x * grad_x + grad_y * grad_y
    return float(numpy.mean(tenengrad)), box


class HillClimbAutofocus:
    def __init__(self, focuser, start_focus: int = 350) -> None:
        self.focuser = focuser
        self.current_focus = clamp_focus(start_focus)
        self.anchor_focus = self.current_focus
        self.anchor_score: float | None = None
        self.current_score: float | None = None
        self.current_roi = None
        self.direction = 1
        self.step = SEARCH_STEP
        self.settle_frames = SETTLE_FRAMES
        self.check_interval = CHECK_INTERVAL
        self.frames_until_ready = SETTLE_FRAMES
        self.frames_since_check = 0
        self.pending_probe: tuple[int, float] | None = None
        self.failed_probes = 0
        self.mode = "search"
        self.action = "starting"
        self.recent_scores = deque(maxlen=SCENE_DROP_COUNT)
        self.low_score_streak = 0
        self._write_focus(self.current_focus)

    def _write_focus(self, value: int) -> None:
        value = clamp_focus(value)
        self.focuser.set(self.focuser.OPT_FOCUS, value)
        self.current_focus = value

    def _schedule_probe(self) -> None:
        probe_focus = clamp_focus(self.current_focus + self.direction * self.step)
        if probe_focus == self.current_focus:
            self.direction *= -1
            probe_focus = clamp_focus(self.current_focus + self.direction * self.step)

        baseline_score = self.current_score if self.current_score is not None else 0.0
        self.pending_probe = (self.current_focus, baseline_score)
        self._write_focus(probe_focus)
        self.frames_until_ready = self.settle_frames
        self.action = f"probe->{probe_focus}"

    def _accept_probe(self, score: float) -> None:
        self.anchor_focus = self.current_focus
        self.anchor_score = score
        self.pending_probe = None
        self.failed_probes = 0
        self.mode = "track"
        if self.step > TRACK_STEP:
            self.step = TRACK_STEP
        self.action = f"accept@{self.current_focus}"
        self._schedule_probe()

    def _reject_probe(self) -> None:
        assert self.pending_probe is not None
        previous_focus, previous_score = self.pending_probe
        self._write_focus(previous_focus)
        self.current_score = previous_score
        self.pending_probe = None
        self.direction *= -1
        self.failed_probes += 1
        self.action = f"reject->{previous_focus}"

        if self.failed_probes >= 3:
            self.step = max(MIN_STEP, self.step // 2)
            self.failed_probes = 0
            self.action += f" step={self.step}"

        self.frames_until_ready = self.settle_frames

    def _reset_search(self, score: float) -> None:
        self.anchor_focus = self.current_focus
        self.anchor_score = score
        self.pending_probe = None
        self.direction = 1
        self.step = SEARCH_STEP
        self.failed_probes = 0
        self.low_score_streak = 0
        self.mode = "search"
        self.action = "reacquire"
        self.frames_until_ready = self.settle_frames

    def _scene_changed(self) -> bool:
        if self.anchor_score is None or len(self.recent_scores) < SCENE_DROP_COUNT:
            return False

        recent_average = sum(self.recent_scores) / len(self.recent_scores)
        if recent_average < self.anchor_score * SCENE_DROP_RATIO:
            self.low_score_streak += 1
        else:
            self.low_score_streak = 0
        return self.low_score_streak >= SCENE_DROP_COUNT

    def update(self, frame) -> dict | None:
        if self.frames_until_ready > 0:
            self.frames_until_ready -= 1
            return None

        self.frames_since_check += 1
        if self.frames_since_check < self.check_interval:
            return None
        self.frames_since_check = 0

        score, roi = sharpness_score(frame)
        self.current_score = score
        self.current_roi = roi
        self.recent_scores.append(score)

        if self.anchor_score is None:
            self.anchor_score = score
            self.anchor_focus = self.current_focus
            self.action = f"anchor@{self.current_focus}"
            self._schedule_probe()
            return self.snapshot()

        if self._scene_changed():
            self._reset_search(score)
            self._schedule_probe()
            return self.snapshot()

        if self.pending_probe is None:
            self._schedule_probe()
            return self.snapshot()

        _, baseline_score = self.pending_probe
        if score > baseline_score * (1.0 + IMPROVE_RATIO):
            self._accept_probe(score)
        else:
            self._reject_probe()

        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "focus": self.current_focus,
            "score": self.current_score,
            "anchor_focus": self.anchor_focus,
            "anchor_score": self.anchor_score,
            "step": self.step,
            "action": self.action,
            "roi": self.current_roi,
        }


def draw_overlay(frame, autofocus: HillClimbAutofocus, fps: float | None) -> None:
    data = autofocus.snapshot()
    lines = [
        "AF: ON",
        f"Mode: {data['mode']}",
        f"Focus: {data['focus']}",
        f"Sharpness: {data['score']:.1f}" if data["score"] is not None else "Sharpness: measuring...",
        f"Step: {data['step']}",
        f"Action: {data['action']}",
        f"FPS: {fps:.1f}" if fps is not None else "FPS: measuring...",
    ]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28

    if data["roi"] is not None:
        x1, y1, x2, y2 = data["roi"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
