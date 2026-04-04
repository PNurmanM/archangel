# ArchAngel — Real-Time Neural Activity Prediction

<p align="center">
  <strong>The world's first real-time brain activity prediction system using edge computing and GPU-accelerated neural encoding.</strong>
</p>

---

## What is ArchAngel?

ArchAngel is a groundbreaking system that predicts human brain activity in real-time from a live camera feed. Using Meta's TRIBE v2 brain encoding model — a 2.5 billion parameter neural network trained on 1,000+ hours of fMRI brain scans from 700+ volunteers — ArchAngel transforms visual input into a full cortical activation map of 20,484 brain vertices, updated multiple times per second.

A camera on an NVIDIA Jetson Orin Nano captures what a person is seeing. Those frames stream to a GPU workstation where V-JEPA2 (Meta's state-of-the-art video understanding model) and a brain mapping network predict how the human brain would respond — including which regions activate for emotion, attention, fear, memory, and cognition. Results flow back in under 600ms and are displayed on a sleek web dashboard with live charts, spike detection, and brain region breakdowns.

**This is not a toy demo.** The predictions are based on the same neuroscience that powers real fMRI research, compressed from months of lab work into computation that runs in milliseconds.

---

## Architecture

```
┌─────────────────────────────┐         ┌─────────────────────────────────────┐
│  NVIDIA Jetson Orin Nano    │         │  GPU Workstation                    │
│                             │         │  (RTX 5090 + RTX 4090)              │
│  Arducam IMX477 Camera      │  TCP    │                                     │
│  30fps 1080p                ├────────►│  V-JEPA2 (ViT-g, 2.5B params)      │
│  GStreamer + NVENC           │  frames │  Wav2Vec-BERT (audio)               │
│                             │         │  TRIBE v2 Brain Model               │
│  Preview + Overlay          │◄────────┤  Starlette + WebSocket              │
│                             │  results│                                     │
└─────────────────────────────┘         └──────────┬──────────────────────────┘
                                                   │ WebSocket
                                                   ▼
                                        ┌─────────────────────┐
                                        │  Next.js Dashboard   │
                                        │  Live brain activity │
                                        │  Emotion detection   │
                                        │  Spike alerts        │
                                        └─────────────────────┘
```

---

## Performance

| Metric | Before Optimization | After Optimization | Speedup |
|--------|---:|---:|---:|
| Video processing | 543 seconds | 5.1 seconds | **107x** |
| Brain predictions/sec | 0.04 FPS | 19.6 FPS | **490x** |
| Model loading | 47 seconds | 4.2 seconds (cached in VRAM) | **11x** |
| Video decoding | 254 seconds (MoviePy) | 0.2 seconds (OpenCV) | **1,270x** |
| End-to-end latency | Minutes | < 600ms | Real-time |

### Key Optimizations
- **BF16 mixed precision** — halved memory bandwidth on the 2.5B parameter V-JEPA2 model
- **SDPA (Scaled Dot-Product Attention)** — hardware-accelerated attention kernels
- **Dual-GPU parallel inference** — RTX 5090 and RTX 4090 process clips simultaneously
- **OpenCV batch decode** — replaced MoviePy's 2,688 random seeks with sequential batch read
- **Models permanently in VRAM** — eliminated the original pipeline's lazy-load-then-delete cycle
- **Sliding window streaming** — temporal clip history enables the brain model to detect changes over time

---

## Emotion & Spike Detection

ArchAngel goes beyond raw brain activation. It tracks emotion-specific brain circuits:

- **Fear/Anxiety** — Insula, Anterior Cingulate Cortex, Parahippocampal regions
- **Anger/Stress** — ACC, Orbitofrontal Cortex, Prefrontal regions
- **Emotional Arousal** — Insula, Temporal Pole, Subcallosal regions
- **Social Emotion** — Superior Temporal Sulcus, Temporal Pole

The system compares current emotional activation against a rolling baseline, with visual cortex activity factored out to prevent motion from triggering false emotional spikes. When emotion regions activate disproportionately to visual processing, a **SPIKE** alert fires.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Edge Device | NVIDIA Jetson Orin Nano Super (8GB) |
| Camera | Arducam IMX477 12MP with motorized autofocus |
| Video Encoder | V-JEPA2 ViT-g (2.5B params, Meta) |
| Audio Encoder | Wav2Vec-BERT 2.0 (Meta) |
| Brain Model | TRIBE v2 (Meta, trained on 700+ fMRI subjects) |
| GPU Inference | RTX 5090 (32GB) + RTX 4090 (24GB) |
| Backend | Python, PyTorch, Starlette, WebSocket |
| Frontend | Next.js 16, React 19, Tailwind CSS, Recharts, Framer Motion |
| Protocol | Custom binary TCP (frames) + WebSocket (predictions) |

---

## Potential Applications

- **Neuromarketing** — Predict viewer emotional response to advertisements, trailers, and product designs without expensive fMRI studies
- **Content Safety** — Real-time detection of content that triggers extreme emotional responses
- **Accessibility** — Brain-computer interface research for individuals with motor disabilities
- **Education** — Measure cognitive engagement and emotional response to learning materials
- **Mental Health** — Monitor stress and anxiety indicators in therapeutic settings
- **Entertainment** — Adaptive media that responds to predicted viewer brain state
- **Security** — Screening systems that detect stress responses to visual stimuli
- **Pharmaceutical Research** — Virtual brain experiments testing neural responses to candidate compounds

---

## Challenges Overcome

1. **543-second inference reduced to 5 seconds** — The original TRIBE v2 pipeline was designed for batch processing on SLURM clusters. We rebuilt it from scratch for real-time streaming, bypassing the entire neuralset data pipeline.

2. **MoviePy's 2,688 random seeks** — The original code extracted each video frame individually via random file seeks. We batch-decoded all frames sequentially with OpenCV — 1,270x faster.

3. **Models deleted after every use** — The original `_free_extractor_model()` function deleted GPU models from VRAM after each extraction to save memory on shared clusters. For real-time inference, we keep everything permanently loaded.

4. **ViT-g on Blackwell architecture** — PyTorch's CUDA 12.1 builds didn't support RTX 5090 (sm_120). We compiled with CUDA 12.8 to unlock both GPUs.

5. **Live emotion detection from visual cortex predictions** — TRIBE v2 predicts cortical surface activation, but key emotion structures (amygdala) are subcortical. We developed a proxy approach using surface-level emotion-adjacent regions with visual baseline subtraction.

6. **WebSocket through Starlette** — FastAPI's CORSMiddleware blocks WebSocket upgrades with HTTP 403. We built a raw Starlette app with manual CORS headers that doesn't interfere with WebSocket connections.

---

## Quick Start

```bash
# PC: Start the inference server + API
python api_server.py --variant vitg --window-size 16 --stride 3

# PC: Start the frontend
cd hhgfront && npm install && npm run build && npx next start -p 3001

# Jetson: Stream camera to PC
python jetson_stream_client.py --host <PC_IP> --send-every 1

# Open http://localhost:3001 and click "Go Live"
```

---

## Credits

- **TRIBE v2** — Meta FAIR ([paper](https://arxiv.org/abs/2507.22229), [code](https://github.com/facebookresearch/tribev2))
- **V-JEPA2** — Meta FAIR ([blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/))
- **Destrieux Atlas** — Nilearn / FreeSurfer

---

<p align="center">
  <em>ArchAngel — See cognition unfold in real time.</em>
</p>
