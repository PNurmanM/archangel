"""
TRIBE v2 Real-Time Pipeline — Optimized for GPU inference.

Bypasses the slow neuralset data pipeline and runs V-JEPA2 + Wav2Vec-BERT
+ brain model directly with:
  - BF16 mixed precision
  - SDPA (scaled dot-product attention)
  - torch.compile on V-JEPA2
  - Batch video frame decoding via OpenCV (no MoviePy seek-per-frame)
  - Batched V-JEPA2 clip processing
  - Dual-GPU: video on cuda:1 (RTX 5090), audio on cuda:0 (RTX 4090)
  - Models kept permanently in VRAM

Usage:
  python tribev2_realtime.py "SpaceX Raptor Engine Firing.mp4"
  python tribev2_realtime.py "SpaceX Raptor Engine Firing.mp4" --visualize
  python tribev2_realtime.py --serve    # interactive mode
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ── Configuration ─────────────────────────────────────────────────

VIDEO_FREQ = 2.0           # brain predictions per second (matches TRIBE config)
CLIP_DURATION = 4.0        # seconds per V-JEPA2 clip
NUM_FRAMES = 64            # frames per clip (V-JEPA2 fpc64)
AUDIO_SR = 16000           # Wav2Vec-BERT sample rate
DURATION_TRS = 100         # brain model temporal window (from TRIBE config)

# Layer selection: [0.5, 0.75, 1.0] with group_mean → 2 output layer groups
VIDEO_LAYERS = [0.5, 0.75, 1.0]
AUDIO_LAYERS = [0.5, 0.75, 1.0]

# Model variants
VJEPA2_MODELS = {
    "vitg": ("facebook/vjepa2-vitg-fpc64-256", 64, 1408),  # (repo, frames, hidden_dim)
    "vitl": ("facebook/vjepa2-vitl-fpc64-256", 64, 1024),  # ~8x fewer params
    "vith": ("facebook/vjepa2-vith-fpc64-256", 64, 1280),  # middle ground
}
BRAIN_VIDEO_DIM = 1408  # brain model was trained with ViT-g features

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ── GPU device selection ──────────────────────────────────────────

def select_devices():
    """Pick best GPUs. Prefer 5090 for video (heavier), 4090 for audio."""
    n = torch.cuda.device_count()
    if n == 0:
        print("WARNING: No CUDA GPUs found, falling back to CPU")
        return "cpu", "cpu"
    if n == 1:
        return "cuda:0", "cuda:0"
    # Find the GPU with most VRAM for video (typically 5090)
    vram = [(torch.cuda.get_device_properties(i).total_memory, i) for i in range(n)]
    vram.sort(reverse=True)
    video_dev = f"cuda:{vram[0][1]}"
    audio_dev = f"cuda:{vram[1][1]}"
    return video_dev, audio_dev


# ── Layer aggregation (matches neuralset group_mean) ──────────────

def aggregate_layers_group_mean(hidden_states, layer_fracs):
    """Replicate neuralset's group_mean layer aggregation.

    hidden_states: tuple of (n_tokens, dim) tensors, one per model layer
    layer_fracs: e.g. [0.5, 0.75, 1.0]
    Returns: (n_groups, dim) where n_groups = len(layer_fracs) - 1
    """
    n_layers = len(hidden_states)
    indices = sorted(set(int(f * (n_layers - 1)) for f in layer_fracs))
    indices[-1] = min(indices[-1] + 1, n_layers)  # make upper bound exclusive
    groups = []
    for l1, l2 in zip(indices[:-1], indices[1:]):
        group = torch.stack([hidden_states[i] for i in range(l1, l2)])
        groups.append(group.mean(dim=0))
    return torch.stack(groups)  # (n_groups, n_tokens, dim)


# ── Video frame extraction (batch OpenCV) ─────────────────────────

def decode_video_opencv(video_path: str):
    """Decode entire video into numpy array using OpenCV. Much faster than
    MoviePy's random-access get_frame() which seeks per frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames), fps, duration


def extract_audio_wav(video_path: str, sr: int = 16000):
    """Extract audio from video. Tries torchaudio → moviepy → soundfile fallback."""
    # Try torchaudio first (fastest if torchcodec/ffmpeg available)
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(video_path)
        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sample_rate, sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), sr
    except Exception:
        pass

    # Fallback: moviepy (already installed by tribev2)
    import tempfile
    from moviepy import VideoFileClip
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        # No audio — return silence
        duration = clip.duration
        return torch.zeros(int(duration * sr)), sr
    tmp = tempfile.mktemp(suffix=".wav")
    clip.audio.write_audiofile(tmp, fps=sr, nbytes=2, codec="pcm_s16le",
                               logger=None)
    clip.close()
    import soundfile as sf
    data, _ = sf.read(tmp, dtype="float32")
    Path(tmp).unlink(missing_ok=True)
    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data), sr


def sample_clip_frames(all_frames: np.ndarray, video_fps: float,
                       clip_end_time: float, clip_duration: float,
                       num_frames: int):
    """Sample num_frames from a clip ending at clip_end_time.
    Returns (num_frames, H, W, 3) numpy array."""
    clip_start = max(0, clip_end_time - clip_duration)
    # Evenly sample within the clip window
    times = np.linspace(clip_start, clip_end_time, num_frames, endpoint=False)
    frame_indices = np.clip((times * video_fps).astype(int), 0, len(all_frames) - 1)
    return all_frames[frame_indices]


# ── Model loading with all optimizations ──────────────────────────

class TRIBEv2Realtime:
    """Holds all three models in VRAM, ready for instant inference."""

    def __init__(self, cache_folder="./cache", compile_video=True, variant="vitg"):
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(exist_ok=True)
        self.video_device, self.audio_device = select_devices()
        self._compile_video = compile_video
        self.variant = variant
        self.vjepa2_repo, self.num_frames, self.vjepa2_hidden = VJEPA2_MODELS[variant]

        print(f"  V-JEPA2 variant: {variant} ({self.vjepa2_repo})")
        print(f"  Video GPU: {self.video_device} ({torch.cuda.get_device_name(int(self.video_device[-1]))})")
        print(f"  Audio GPU: {self.audio_device} ({torch.cuda.get_device_name(int(self.audio_device[-1]))})")
        print()

        t0 = time.time()
        self._load_video_model()
        self._load_audio_model()
        self._load_brain_model()
        dt = time.time() - t0
        print(f"\n  All models loaded in {dt:.1f}s — ready for real-time inference")

    def _load_video_model(self):
        """Load V-JEPA2 with bf16 + SDPA on ALL available GPUs."""
        print(f"  Loading V-JEPA2 ({self.variant}, {self.vjepa2_repo})...")
        t0 = time.time()
        from transformers import AutoModel, AutoVideoProcessor

        self.video_processor = AutoVideoProcessor.from_pretrained(self.vjepa2_repo)

        # Load on all GPUs for parallel processing
        self.video_models = []
        self.video_devices = []

        for dev in [self.video_device, self.audio_device]:
            m = AutoModel.from_pretrained(
                self.vjepa2_repo,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                output_hidden_states=True,
            ).to(dev).eval()
            if self._compile_video:
                m = torch.compile(m, mode="default")
            self.video_models.append(m)
            self.video_devices.append(dev)

        self.video_model = self.video_models[0]
        n_gpus = len(self.video_models)

        if self._compile_video:
            print(f"    Compiled on {n_gpus} GPUs (warmup on first call)...")

        dt = time.time() - t0
        print(f"    V-JEPA2 ready on {n_gpus} GPUs in {dt:.1f}s")

    def _load_audio_model(self):
        """Load Wav2Vec-BERT with bf16."""
        print("  Loading Wav2Vec-BERT...")
        t0 = time.time()
        from transformers import AutoModel, AutoFeatureExtractor

        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.audio_model = AutoModel.from_pretrained(
            "facebook/w2v-bert-2.0",
            dtype=torch.bfloat16,
        ).to(self.audio_device)
        self.audio_model.eval()
        dt = time.time() - t0
        print(f"    Wav2Vec-BERT ready in {dt:.1f}s")

    def _load_brain_model(self):
        """Load TRIBE v2 brain prediction model from checkpoint."""
        print("  Loading TRIBE v2 brain model...")
        t0 = time.time()

        from huggingface_hub import hf_hub_download
        import yaml
        from exca import ConfDict

        config_path = hf_hub_download("facebook/tribev2", "config.yaml")
        ckpt_path = hf_hub_download("facebook/tribev2", "best.ckpt")

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        build_args = ckpt["model_build_args"]
        state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
        del ckpt

        # Build model with average_subjects=True (matches checkpoint)
        # We need to import tribev2 to register the model classes
        from tribev2.model import FmriEncoder
        bmc = config["brain_model_config"]
        # Set average_subjects mode — checkpoint was saved with averaged weights
        bmc["subject_layers"]["average_subjects"] = True
        bmc["subject_layers"]["n_subjects"] = 0
        brain_config = FmriEncoder(**bmc)
        self.brain_model = brain_config.build(**build_args)
        self.brain_model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict

        # Brain model is small — put on video device
        self.brain_device = self.video_device
        self.brain_model.to(self.brain_device)
        self.brain_model.eval()
        self.brain_build_args = build_args

        dt = time.time() - t0
        print(f"    Brain model ready in {dt:.1f}s")

    # ── Feature extraction ────────────────────────────────────────

    def _process_clip_on_gpu(self, clip_frames, gpu_idx):
        """Process a single clip on a specific GPU. Returns (n_groups, dim) tensor on CPU."""
        dev = self.video_devices[gpu_idx]
        model = self.video_models[gpu_idx]

        inputs = self.video_processor([clip_frames], return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        per_layer = [hs[0] for hs in hidden_states]
        per_layer_pooled = [layer.mean(dim=0) for layer in per_layer]
        grouped = aggregate_layers_group_mean(per_layer_pooled, VIDEO_LAYERS)
        # (n_groups, hidden_dim)

        # Pad to brain model's expected dim if using smaller variant
        if self.vjepa2_hidden < BRAIN_VIDEO_DIM:
            pad_size = BRAIN_VIDEO_DIM - self.vjepa2_hidden
            grouped = F.pad(grouped, (0, pad_size))  # pad last dim

        return grouped.float().cpu()

    def extract_video_features(self, all_frames: np.ndarray, video_fps: float,
                               video_duration: float, batch_size: int = 2):
        """Extract V-JEPA2 features using all available GPUs in parallel."""
        n_timesteps = int(video_duration * VIDEO_FREQ)
        clip_end_times = np.linspace(0, video_duration, n_timesteps + 1)[1:]
        n_gpus = len(self.video_models)

        # Pre-sample all clips (CPU work, fast)
        all_clips = []
        for i in range(n_timesteps):
            all_clips.append(sample_clip_frames(
                all_frames, video_fps, clip_end_times[i],
                CLIP_DURATION, self.num_frames
            ))

        print(f"    Processing {n_timesteps} clips across {n_gpus} GPUs...")
        t0 = time.time()

        # Process clips round-robin across GPUs using threads
        all_features = [None] * n_timesteps
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_clip(idx):
            gpu_idx = idx % n_gpus
            return idx, self._process_clip_on_gpu(all_clips[idx], gpu_idx)

        # Use n_gpus threads so each GPU gets one concurrent request
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = []
            done_count = 0
            # Submit in batches of n_gpus to keep all GPUs busy
            for i in range(n_timesteps):
                futures.append(executor.submit(process_clip, i))

            for future in as_completed(futures):
                idx, feat = future.result()
                all_features[idx] = feat
                done_count += 1
                elapsed = time.time() - t0
                fps = done_count / elapsed if elapsed > 0 else 0
                print(f"      {done_count}/{n_timesteps} clips | {fps:.1f} clips/s | "
                      f"{fps * CLIP_DURATION:.1f} video-sec/s", end="\r")

        print()
        dt = time.time() - t0
        effective_fps = (n_timesteps * self.num_frames) / dt
        print(f"    V-JEPA2 encoding: {dt:.1f}s ({effective_fps:.0f} frames/s, "
              f"{n_timesteps/dt:.1f} clips/s)")

        features = torch.stack(all_features)
        features = features.permute(1, 2, 0)
        return features

    def extract_audio_features(self, waveform: torch.Tensor, sr: int,
                               video_duration: float):
        """Extract Wav2Vec-BERT features. Returns (n_groups, dim, n_timesteps)."""
        n_timesteps = int(video_duration * VIDEO_FREQ)

        print(f"    Processing {video_duration:.1f}s audio...")
        t0 = time.time()

        # Process full audio at once
        inputs = self.audio_feature_extractor(
            waveform.numpy(), sampling_rate=sr,
            return_tensors="pt", do_normalize=True,
        )
        input_key = "input_features" if "input_features" in inputs else "input_values"
        input_tensor = inputs[input_key].to(self.audio_device)

        with torch.inference_mode(), torch.autocast(self.audio_device.split(":")[0], dtype=torch.bfloat16):
            outputs = self.audio_model(input_tensor, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, tokens, dim)
        n_model_layers = len(hidden_states)

        # Stack all layers: (n_layers, tokens, dim)
        all_layers = torch.stack([hs.squeeze(0) for hs in hidden_states])

        # Token aggregation: mean over tokens → (n_layers, dim)
        all_layers_pooled = all_layers.mean(dim=1)

        # Layer aggregation: group_mean
        indices = sorted(set(int(f * (n_model_layers - 1)) for f in AUDIO_LAYERS))
        indices[-1] = min(indices[-1] + 1, n_model_layers)
        groups = []
        for l1, l2 in zip(indices[:-1], indices[1:]):
            groups.append(all_layers_pooled[l1:l2].mean(dim=0))
        grouped = torch.stack(groups).float().cpu()  # (n_groups, dim)

        # Resample to match video timesteps
        # Audio produces one feature vector for the whole clip; we need n_timesteps
        # Repeat to fill temporal dimension (TRIBE expects (n_groups, dim, T))
        features = grouped.unsqueeze(-1).expand(-1, -1, n_timesteps)

        dt = time.time() - t0
        print(f"    Wav2Vec-BERT encoding: {dt:.1f}s")
        return features

    # ── Brain prediction ──────────────────────────────────────────

    def predict_brain(self, video_features, audio_features):
        """Run the brain model on extracted features.
        Returns (n_kept_timesteps, 20484) numpy array."""
        from neuralset.dataloader import SegmentData
        import neuralset.segments as nsseg

        # Build a fake SegmentData batch matching what the brain model expects
        # The model's aggregate_features reads batch.data[modality] with shape (B, L, D, T)
        # and batch.data["subject_id"]

        T = video_features.shape[-1]  # number of timesteps

        # Reshape to (1, n_groups, dim, T) — batch size 1
        video_feat = video_features.unsqueeze(0).to(self.brain_device)
        audio_feat = audio_features.unsqueeze(0).to(self.brain_device)

        # Build minimal SegmentData
        data = {
            "video": video_feat,
            "audio": audio_feat,
            "subject_id": torch.zeros(1, dtype=torch.long, device=self.brain_device),
        }

        # Create a simple namespace that has .data attribute
        class FakeBatch:
            pass
        batch = FakeBatch()
        batch.data = data

        print(f"    Running brain model (T={T})...")
        t0 = time.time()
        with torch.inference_mode():
            y_pred = self.brain_model(batch).detach().cpu().numpy()
        # y_pred shape: (1, n_outputs, n_output_timesteps)
        # Rearrange to (n_timesteps, n_outputs)
        from einops import rearrange
        preds = rearrange(y_pred, "b d t -> (b t) d")

        # Remove empty segments (segments with no content)
        # In the original pipeline, only segments with events are kept
        # For simplicity, keep all non-zero predictions
        keep = np.abs(preds).sum(axis=1) > 1e-6
        preds = preds[keep]

        dt = time.time() - t0
        print(f"    Brain prediction: {dt:.2f}s ({preds.shape[0]} timesteps)")
        return preds

    # ── Full pipeline ─────────────────────────────────────────────

    def process_video(self, video_path: str, output_dir: str = "./outputs",
                      visualize: bool = False, save_frames: bool = False,
                      make_movie: bool = True, movie_fps: int = 2,
                      surface: str = "inflated", html: bool = False,
                      batch_size: int = 2):
        """Process a video end-to-end and return brain predictions."""
        input_stem = Path(video_path).stem
        out_dir = Path(output_dir) / input_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        t_total = time.time()

        # 1. Decode video frames (OpenCV batch decode)
        print("\n[1/4] Decoding video frames (OpenCV)...")
        t0 = time.time()
        all_frames, video_fps, video_duration = decode_video_opencv(video_path)
        dt = time.time() - t0
        print(f"    {len(all_frames)} frames decoded in {dt:.2f}s "
              f"({len(all_frames)/dt:.0f} fps decode speed)")

        # 2. Extract features (video on ALL GPUs, then audio)
        print("\n[2/4] Extracting features...")

        # Video on both GPUs (the heavy part)
        video_features = self.extract_video_features(all_frames, video_fps, video_duration,
                                                      batch_size=batch_size)
        del all_frames  # free RAM

        # Audio (fast, ~0.2s)
        try:
            waveform, sr = extract_audio_wav(video_path)
            audio_features = self.extract_audio_features(waveform, sr, video_duration)
        except Exception as e:
            print(f"    Audio extraction failed: {e} — using zero features")
            n_timesteps = video_features.shape[-1]
            audio_features = torch.zeros(2, 1024, n_timesteps)

        # 3. Brain prediction
        print("\n[3/4] Predicting brain activity...")
        preds = self.predict_brain(video_features, audio_features)

        # 4. Summary
        dt_total = time.time() - t_total
        realtime_ratio = video_duration / dt_total
        effective_fps = (preds.shape[0]) / dt_total

        print("\n" + "=" * 60)
        print("PERFORMANCE")
        print("=" * 60)
        print(f"  Video duration    : {video_duration:.1f}s")
        print(f"  Processing time   : {dt_total:.1f}s")
        print(f"  Real-time ratio   : {realtime_ratio:.2f}x")
        print(f"  Brain pred FPS    : {effective_fps:.1f}")
        print(f"  Predictions shape : {preds.shape}")
        print(f"  Value range       : [{preds.min():.4f}, {preds.max():.4f}]")

        # Import summary function from original script
        print("\n" + "=" * 60)
        print("BRAIN ACTIVITY SUMMARY")
        print("=" * 60)
        try:
            from tribev2_explore import get_brain_summary
            summary = get_brain_summary(preds)
            print(summary)
        except ImportError:
            summary = f"Predictions shape: {preds.shape}\nValue range: [{preds.min():.4f}, {preds.max():.4f}]"
            print(summary)

        # Save outputs
        npy_path = out_dir / f"{input_stem}_brain_preds.npy"
        np.save(str(npy_path), preds)
        print(f"  Raw predictions saved to: {npy_path}")

        summary_path = out_dir / f"{input_stem}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"  Summary saved to: {summary_path}")

        # Visualization
        if visualize:
            print("\n[4/4] Generating brain visualizations...")
            try:
                from brain_visualizer import visualize_all
                visualize_all(
                    preds=preds, output_dir=str(out_dir),
                    image_stem=input_stem, surface=surface,
                    save_frames=save_frames, make_movie=make_movie,
                    movie_fps=movie_fps, html=html,
                )
            except Exception as e:
                print(f"  Visualization failed: {e}")

        print(f"\n  All outputs saved to: {out_dir}")
        print(f"  Total time: {dt_total:.1f}s (real-time ratio: {realtime_ratio:.2f}x)")
        return preds


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TRIBE v2 Real-Time Brain Activity Explorer"
    )
    parser.add_argument("input", nargs="?", help="Path to a video file")
    parser.add_argument("--cache", default="./cache")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--no-movie", action="store_true")
    parser.add_argument("--movie-fps", type=int, default=2)
    parser.add_argument("--surface", default="inflated",
                        choices=["inflated", "pial", "white"])
    parser.add_argument("--serve", action="store_true",
                        help="Server mode: load once, process many")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (faster startup, slower inference)")
    parser.add_argument("--fast", action="store_true",
                        help="Use ViT-L/fpc16 (smaller, faster, ~slight accuracy loss)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="V-JEPA2 clip batch size (default: 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("TRIBE v2 — REAL-TIME MODE")
    print("=" * 60)
    print()

    # Load all models
    variant = "vitl" if args.fast else "vitg"
    engine = TRIBEv2Realtime(
        cache_folder=args.cache,
        compile_video=not args.no_compile,
        variant=variant,
    )

    if args.serve:
        print("\nReady! Paste a video path and press Enter.")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down.")
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break
            file_path = user_input.strip("\"'")
            if not Path(file_path).exists():
                print(f"  File not found: {file_path}")
                continue
            if Path(file_path).suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"  Unsupported format: {Path(file_path).suffix}")
                continue
            try:
                engine.process_video(
                    file_path,
                    output_dir=args.output_dir,
                    visualize=args.visualize,
                    save_frames=args.save_frames,
                    make_movie=not args.no_movie,
                    movie_fps=args.movie_fps,
                    surface=args.surface,
                    html=args.html,
                    batch_size=args.batch_size,
                )
            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                traceback.print_exc()
            print()
    else:
        if not args.input:
            parser.error("the following arguments are required: input (or use --serve)")
        if not Path(args.input).exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)
        engine.process_video(
            args.input,
            output_dir=args.output_dir,
            visualize=args.visualize,
            save_frames=args.save_frames,
            make_movie=not args.no_movie,
            movie_fps=args.movie_fps,
            surface=args.surface,
            html=args.html,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
