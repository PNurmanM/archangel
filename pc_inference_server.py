#!/usr/bin/env python3
"""
TRIBE v2 Real-Time Inference Server

Receives JPEG frames from Jetson, maintains a sliding window, and runs
brain activity prediction using the optimized TRIBE v2 pipeline.

Architecture:
  Jetson (30fps camera) → TCP frames → this server → TRIBE v2 → results back

Sliding window strategy:
  - Accumulate frames into a window (default 30 = 1s at 30fps)
  - Every N new frames (default 5), run V-JEPA2 on the current window
  - This gives ~6 brain predictions per second at 30fps input
  - The window slides: old frames drop off, new ones come in

Usage:
  python pc_inference_server.py                          # default ViT-L (fast)
  python pc_inference_server.py --variant vitg            # full accuracy
  python pc_inference_server.py --window-size 60 --stride 10  # 2s window, predict every 10 frames
"""

from __future__ import annotations

import argparse
import socket
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stream_protocol import recv_message, send_message


# ── Import TRIBE v2 pipeline components ───────────────────────────

from tribev2_realtime import (
    TRIBEv2Realtime,
    aggregate_layers_group_mean,
    VIDEO_LAYERS,
    BRAIN_VIDEO_DIM,
)


# ── Brain region summary (lightweight version for streaming) ──────

_ATLAS_CACHE = {}

# Emotion/stress-related brain regions (Destrieux atlas names)
# These are the regions that light up for fear, stress, anger, arousal
EMOTION_REGIONS = {
    # Insula — visceral emotion, gut feelings, anxiety, disgust
    "G_insular_short": "insula",
    "G_Ins_lg_and_S_cent_ins": "insula",
    # Cingulate cortex — emotional regulation, pain, conflict/stress
    "G_and_S_cingul-Ant": "acc",           # anterior cingulate (stress, anxiety)
    "G_and_S_cingul-Mid-Ant": "acc",       # mid-anterior cingulate
    "G_and_S_cingul-Mid-Post": "pcc",      # posterior cingulate
    "G_cingul-Post-dorsal": "pcc",
    "G_cingul-Post-ventral": "pcc",
    # Orbitofrontal — emotional regulation, anger, reward/punishment
    "G_front_inf-Orbital": "ofc",
    "G_orbital": "ofc",
    "S_orbital_lateral": "ofc",
    "S_orbital-H_Shaped": "ofc",
    "S_orbital_med-olfact": "ofc",
    # Temporal pole — social emotion, fear memory
    "Pole_temporal": "temporal_pole",
    # Parahippocampal — fear conditioning, emotional memory
    "G_oc-temp_med-Parahip": "parahipp",
    # Subcallosal — deep emotional processing
    "G_subcallosal": "subcallosal",
    # Amygdala adjacent (surface proxy — amygdala is subcortical but nearby cortex reflects it)
    "S_collat_transv_ant": "amygdala_adj",
    # Temporal — social/emotional perception
    "G_temp_sup-Lateral": "sts",           # social cues, voice emotion
    "S_temporal_sup": "sts",               # theory of mind, empathy
    # Frontal — emotional control, stress response
    "G_front_inf-Triangul": "vlpfc",       # emotion regulation
    "G_front_middle": "dlpfc",             # cognitive control under stress
}

# Group into emotion systems
EMOTION_SYSTEMS = {
    "fear_anxiety": ["insula", "acc", "amygdala_adj", "parahipp"],
    "emotional_arousal": ["insula", "acc", "temporal_pole", "subcallosal"],
    "anger_stress": ["acc", "ofc", "dlpfc", "vlpfc", "insula"],
    "social_emotion": ["sts", "temporal_pole", "ofc"],
}


def _get_atlas():
    """Load surface atlas once, precompute region masks."""
    if not _ATLAS_CACHE:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from nilearn import datasets
            atlas = datasets.fetch_atlas_surf_destrieux(verbose=0)

        labels_lh = atlas["map_left"]
        labels_rh = atlas["map_right"]
        label_names = atlas["labels"]
        n_vert = len(labels_lh)

        # Precompute masks for ALL regions
        region_masks = {}
        for label_idx, label_name in enumerate(label_names):
            name = label_name if isinstance(label_name, str) else label_name.decode()
            if name in ("Unknown", "Medial_wall"):
                continue
            mask_lh = labels_lh == label_idx
            mask_rh = labels_rh == label_idx
            if mask_lh.sum() + mask_rh.sum() == 0:
                continue
            # Combined mask across both hemispheres (in vertex space)
            full_mask = np.zeros(n_vert * 2, dtype=bool)
            full_mask[:n_vert][mask_lh] = True
            full_mask[n_vert:][mask_rh] = True
            region_masks[name] = full_mask

        # Precompute emotion system masks
        emotion_masks = {}
        for system_name, region_types in EMOTION_SYSTEMS.items():
            system_mask = np.zeros(n_vert * 2, dtype=bool)
            for atlas_name, region_type in EMOTION_REGIONS.items():
                if region_type in region_types and atlas_name in region_masks:
                    system_mask |= region_masks[atlas_name]
            emotion_masks[system_name] = system_mask

        from tribev2_explore import REGION_INFO
        _ATLAS_CACHE["region_masks"] = region_masks
        _ATLAS_CACHE["emotion_masks"] = emotion_masks
        _ATLAS_CACHE["region_info"] = REGION_INFO

    return _ATLAS_CACHE


def _get_visual_mask():
    """Build a mask for visual cortex regions to use as baseline."""
    cache = _get_atlas()
    if "visual_mask" not in cache:
        VISUAL_REGIONS = {
            "G_cuneus", "G_occipital_middle", "G_occipital_sup",
            "Pole_occipital", "S_calcarine", "G_and_S_occipital_inf",
            "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal",
        }
        mask = np.zeros_like(list(cache["region_masks"].values())[0], dtype=bool)
        for name in VISUAL_REGIONS:
            if name in cache["region_masks"]:
                mask |= cache["region_masks"][name]
        cache["visual_mask"] = mask
    return cache["visual_mask"]


def get_emotion_scores(preds: np.ndarray) -> dict:
    """Score emotion-specific brain systems.

    Returns raw absolute activation values for each emotion system.
    These are small numbers (typically 0.01-0.10) — the important thing
    is how they CHANGE over time, not their absolute value.
    """
    try:
        cache = _get_atlas()
        mean_act = preds.mean(axis=0)

        scores = {}
        for system_name, mask in cache["emotion_masks"].items():
            if mask.sum() > 0:
                scores[system_name] = float(np.abs(mean_act[mask]).mean())

        # Also get visual for reference
        visual_mask = _get_visual_mask()
        if visual_mask.sum() > 0:
            scores["visual"] = float(np.abs(mean_act[visual_mask]).mean())

        return scores
    except Exception:
        return {}


def get_top_regions(preds: np.ndarray, top_k: int = 5, emotion_only: bool = False) -> list[dict]:
    """Get top-K most active brain regions."""
    try:
        cache = _get_atlas()
        REGION_INFO = cache["region_info"]
        mean_act = preds.mean(axis=0)

        region_scores = {}
        for name, mask in cache["region_masks"].items():
            if emotion_only and name not in EMOTION_REGIONS:
                continue
            if mask.sum() > 0:
                region_scores[name] = float(mean_act[mask].mean())

        sorted_regions = sorted(region_scores.items(), key=lambda x: abs(x[1]), reverse=True)

        results = []
        for name, score in sorted_regions[:top_k]:
            info = REGION_INFO.get(name, (name, "unknown", "", ""))
            results.append({
                "region": info[0],
                "category": info[1],
                "score": round(float(score), 4),
                "description": info[3],
            })
        return results
    except Exception as e:
        return [{"region": "error", "score": 0, "description": str(e)}]


# ── Inference engine (wraps TRIBEv2Realtime for streaming) ────────

class StreamingBrainEngine:
    """Wraps the TRIBE v2 pipeline for frame-by-frame streaming inference.

    Accumulates clip features over time in a sliding history buffer.
    The brain model needs temporal context (T>=10) to produce meaningful
    variation — a single snapshot always gives ~the same engagement score.
    """

    def _encode_single_clip(self, frames: np.ndarray, gpu_idx: int = 0):
        """Run V-JEPA2 on a single clip of frames. Returns (n_groups, dim) tensor."""
        dev = self.engine.video_devices[gpu_idx]
        model = self.engine.video_models[gpu_idx]
        proc = self.engine.video_processor

        inputs = proc([frames], return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        per_layer = [hs[0] for hs in hidden_states]
        per_layer_pooled = [layer.mean(dim=0) for layer in per_layer]
        grouped = aggregate_layers_group_mean(per_layer_pooled, VIDEO_LAYERS)

        # Pad to brain model dim if needed
        if self.engine.vjepa2_hidden < BRAIN_VIDEO_DIM:
            pad_size = BRAIN_VIDEO_DIM - self.engine.vjepa2_hidden
            grouped = F.pad(grouped, (0, pad_size))

        return grouped.float().cpu()

    def __init__(self, variant: str = "vitl", compile_video: bool = False):
        print("=" * 60)
        print("TRIBE v2 — STREAMING INFERENCE ENGINE")
        print("=" * 60)
        print()

        self.engine = TRIBEv2Realtime(
            cache_folder="./cache",
            compile_video=compile_video,
            variant=variant,
        )

        # Temporal buffer: accumulate clip features over time
        # Brain model needs T>=10 to produce meaningful temporal variation
        self.clip_history: deque[torch.Tensor] = deque(maxlen=15)

        # Warmup
        print("\n  Warming up GPU...")
        t0 = time.time()
        dummy_frames = np.random.randint(0, 255,
            (self.engine.num_frames, 256, 256, 3), dtype=np.uint8)
        _ = self._encode_single_clip(dummy_frames, gpu_idx=0)
        print(f"  Warmup done in {time.time() - t0:.1f}s")
        print()

    def encode_window(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Encode a window of frames into a single clip feature vector.
        Returns (n_groups, dim) tensor."""
        n_frames = self.engine.num_frames
        n_input = len(frames)
        indices = np.linspace(0, n_input - 1, n_frames, dtype=int)
        clip = np.array([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in indices])
        return self._encode_single_clip(clip)

    def predict_from_history(self) -> dict:
        """Run brain model on accumulated clip history.

        The brain model was trained on temporal sequences (T=40+).
        We accumulate clip features over time so the model sees
        how the visual input changes, not just a single snapshot.
        """
        t_start = time.time()
        from einops import rearrange

        T = len(self.clip_history)

        # Stack clip history: list of (n_groups, dim) → (n_groups, dim, T)
        video_feat = torch.stack(list(self.clip_history), dim=-1)  # (n_groups, dim, T)
        audio_feat = torch.zeros(2, 1024, T)

        # Brain model prediction
        t_brain = time.time()
        video_feat_b = video_feat.unsqueeze(0).to(self.engine.brain_device)
        audio_feat_b = audio_feat.unsqueeze(0).to(self.engine.brain_device)

        class FakeBatch:
            pass
        batch = FakeBatch()
        batch.data = {
            "video": video_feat_b,
            "audio": audio_feat_b,
            "subject_id": torch.zeros(1, dtype=torch.long, device=self.engine.brain_device),
        }

        with torch.inference_mode():
            y_pred = self.engine.brain_model(batch).detach().cpu().numpy()

        preds = rearrange(y_pred, "b d t -> (b t) d")
        dt_brain = time.time() - t_brain

        # Split: "now" (last 3) vs "baseline" (rest)
        now_n = min(3, len(preds))
        now_preds = preds[-now_n:]
        baseline_preds = preds[:-now_n] if len(preds) > 5 else preds

        # Emotion scores for NOW vs BASELINE
        now_emotions = get_emotion_scores(now_preds)
        baseline_emotions = get_emotion_scores(baseline_preds)

        # Composite emotion score (weighted toward fear/stress)
        # Compare emotion-only, exclude visual from composite
        weights = {"fear_anxiety": 3.0, "anger_stress": 2.0, "emotional_arousal": 2.0, "social_emotion": 1.0}
        now_composite = sum(now_emotions.get(k, 0) * w for k, w in weights.items() if k in now_emotions) / sum(weights.values())
        base_composite = sum(baseline_emotions.get(k, 0) * w for k, w in weights.items() if k in baseline_emotions) / sum(weights.values())

        # Also compute visual change separately so we can tell motion from emotion
        now_vis = now_emotions.get("visual", 0.01)
        base_vis = baseline_emotions.get("visual", 0.01)
        vis_change_pct = ((now_vis - base_vis) / max(base_vis, 0.001)) * 100

        # Spike detection: emotion change vs visual change
        delta = now_composite - base_composite
        emo_change_pct = (delta / max(base_composite, 0.001)) * 100

        # Key insight: if emotion rises MORE than visual, it's genuine emotion
        # If both rise equally, it's just motion/stimulation
        if abs(vis_change_pct) > 5:
            # Visual is also changing — subtract visual contribution
            net_emo_pct = emo_change_pct - vis_change_pct * 0.5
        else:
            net_emo_pct = emo_change_pct

        spike_pct = net_emo_pct
        if abs(spike_pct) > 25:
            spike_label = "SPIKE!" if spike_pct > 0 else "DROP!"
        elif abs(spike_pct) > 12:
            spike_label = "rising" if spike_pct > 0 else "falling"
        else:
            spike_label = "steady"

        # Top emotion regions only
        top_regions = get_top_regions(now_preds, top_k=5, emotion_only=True)

        dt_total = time.time() - t_start

        return {
            "history_length": T,
            "emotion_score": round(now_composite, 4),
            "baseline_score": round(base_composite, 4),
            "delta": round(delta, 4),
            "spike_pct": round(spike_pct, 1),
            "spike": spike_label,
            "vis_change_pct": round(vis_change_pct, 1),
            "emotions": {k: round(v, 4) for k, v in now_emotions.items()},
            "top_regions": top_regions,
            "timing": {
                "brain_ms": round(dt_brain * 1000, 1),
                "total_ms": round(dt_total * 1000, 1),
            },
        }


# ── Server ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRIBE v2 streaming inference server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--window-size", type=int, default=30,
                        help="Number of frames in sliding window (default: 30 = 1s at 30fps)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Run inference every N new frames (default: 5 = ~6 predictions/s at 30fps)")
    parser.add_argument("--variant", default="vitl", choices=["vitg", "vitl", "vith"],
                        help="V-JEPA2 model variant (default: vitl for speed)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (slower startup, faster steady-state)")
    return parser.parse_args()


def serve_client(client: socket.socket, engine: StreamingBrainEngine,
                 args: argparse.Namespace) -> None:
    """Handle one client connection with sliding window inference."""
    window: deque[np.ndarray] = deque(maxlen=args.window_size)
    engine.clip_history.clear()  # reset temporal history for new client
    frames_since_last_inference = 0
    inference_count = 0
    total_frames = 0
    t_connect = time.time()
    save_dir = Path("testimg")
    save_dir.mkdir(exist_ok=True)

    print(f"  Window: {args.window_size} frames, stride: {args.stride}")
    print(f"  At 30fps input: ~{30 / args.stride:.0f} brain predictions/s")
    print()

    while True:
        try:
            message, payload = recv_message(client)
        except ConnectionError:
            dt = time.time() - t_connect
            print(f"\n  Client disconnected after {dt:.0f}s, {inference_count} inferences")
            return

        if message.get("type") != "frame":
            continue

        # Decode and add to window
        array = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        window.append(frame)
        frames_since_last_inference += 1
        total_frames += 1

        # Save every 100th frame to testimg/
        if total_frames % 100 == 0:
            cv2.imwrite(str(save_dir / f"frame_{total_frames:06d}.jpg"), frame)
            # Keep only last 5 saved images
            saved = sorted(save_dir.glob("frame_*.jpg"))
            for old in saved[:-5]:
                old.unlink()

        # Check if we should run inference
        if len(window) < args.window_size:
            # Window not full yet — still buffering
            send_message(client, {
                "type": "result",
                "status": "buffering",
                "window_fill": len(window),
                "window_size": args.window_size,
                "frame_id": message["frame_id"],
            })
            continue

        if frames_since_last_inference < args.stride:
            # Not at stride yet — skip
            continue

        # ── Sliding window inference ──
        # 1. Encode current 30-frame window into one clip feature
        frames_since_last_inference = 0
        inference_count += 1

        t_enc = time.time()
        clip_feat = engine.encode_window(list(window))
        dt_enc = time.time() - t_enc

        # 2. Add to temporal history (brain model needs multiple clips over time)
        engine.clip_history.append(clip_feat)

        # 3. Run brain model on full clip history
        result = engine.predict_from_history()
        result["timing"]["encode_ms"] = round(dt_enc * 1000, 1)

        send_message(client, {
            "type": "result",
            "status": "ok",
            "frame_id": message["frame_id"],
            "window_size": len(window),
            "inference_count": inference_count,
            **result,
        })

        # Log
        t = result["timing"]
        top = result["top_regions"][0] if result["top_regions"] else {}
        region_name = top.get("region", "?")
        emo = result["emotion_score"]
        spike = result["spike"]
        spike_pct = result["spike_pct"]
        hist = result["history_length"]
        emotions = result.get("emotions", {})
        fear = emotions.get("fear_anxiety", 0)
        anger = emotions.get("anger_stress", 0)
        arousal = emotions.get("emotional_arousal", 0)
        vis = emotions.get("visual", 0)
        vis_chg = result.get("vis_change_pct", 0)
        spike_flag = " ***" if spike in ("SPIKE!", "DROP!") else ""
        print(f"  #{inference_count:4d} | {t['encode_ms']:.0f}ms | T={hist:2d} | "
              f"fear={fear:.4f} anger={anger:.4f} arousal={arousal:.4f} vis={vis:.4f} "
              f"[emo {spike:7s} {spike_pct:+5.1f}% | vis {vis_chg:+5.1f}%] {region_name}{spike_flag}")


def main() -> int:
    args = parse_args()

    # Load models
    engine = StreamingBrainEngine(
        variant=args.variant,
        compile_video=args.compile,
    )

    print(f"Listening on {args.host}:{args.port}")
    print(f"Ready for Jetson connections.\n")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)

        while True:
            client, address = server.accept()
            print(f"Client connected: {address[0]}:{address[1]}")
            with client:
                try:
                    serve_client(client, engine, args)
                except Exception as e:
                    print(f"\n  Error: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
