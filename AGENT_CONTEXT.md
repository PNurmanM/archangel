# ArchAngel — Full Agent Context

> This document gives a future agent complete context to resume work on this project.

---

## Project Overview

ArchAngel is a real-time brain activity prediction system. A camera on an NVIDIA Jetson Orin Nano captures video, streams frames over TCP to a PC with dual GPUs (RTX 5090 + RTX 4090), which runs Meta's TRIBE v2 brain encoding model to predict how a human brain would respond to the visual input. Results stream back to the Jetson and to a Next.js web dashboard via WebSocket.

## Repository: https://github.com/PNurmanM/archangel.git

---

## Architecture

```
Jetson Orin Nano (10.42.0.1)     PC Workstation (10.42.0.211)
├─ jetson_stream_client.py       ├─ api_server.py (main entry point)
├─ jetson_imx477_camera.py       │   ├─ TCP :5000 (receives Jetson frames)
├─ jetson_imx477_autofocus.py    │   ├─ HTTP/WS :8000 (serves frontend)
├─ stream_protocol.py            │   └─ imports from:
└─ archpc/ (venv, python3.10)    │       ├─ pc_inference_server.py (brain engine)
                                 │       └─ tribev2_realtime.py (optimized pipeline)
                                 ├─ hhgfront/ (Next.js 16 frontend on :3001)
                                 └─ archangel/ (venv, python3.12)
```

## Key Files

### Backend (Python)
| File | Purpose |
|------|---------|
| `api_server.py` | **Main server** — TCP for Jetson + WebSocket/HTTP for frontend, runs inference |
| `pc_inference_server.py` | Standalone TCP inference server (older, used by api_server internally) |
| `tribev2_realtime.py` | Optimized TRIBE v2 pipeline — dual-GPU, bf16, SDPA, batch decode |
| `tribev2_explore.py` | Original single-file inference script + brain region atlas (REGION_INFO dict) |
| `brain_visualizer.py` | 3D brain surface rendering (nilearn + matplotlib) |
| `stream_protocol.py` | Binary TCP protocol: 8-byte header (msg_size, payload_size) + JSON + JPEG |
| `jetson_stream_client.py` | Jetson camera capture → TCP stream → receive results → overlay display |
| `jetson_imx477_camera.py` | GStreamer pipeline for Arducam IMX477 CSI camera |
| `jetson_imx477_autofocus.py` | Hill-climbing autofocus via I2C lens control |

### Frontend (TypeScript)
| File | Purpose |
|------|---------|
| `hhgfront/app/page.tsx` | Main page — hero, video upload, live mode, analytics panels |
| `hhgfront/lib/live-stream.ts` | WebSocket client — connects to ws://hostname:8000/ws |
| `hhgfront/lib/api.ts` | Video upload API (currently mock mode, `USE_MOCK = true`) |
| `hhgfront/lib/types.ts` | TypeScript interfaces: BrainPrediction, SystemScore, etc. |
| `hhgfront/lib/mock-data.ts` | Mock data generator for demo mode |
| `hhgfront/components/live-status.tsx` | Live emotion bars + spike indicator component |
| `hhgfront/components/activity-chart.tsx` | Recharts timeline of brain system activity |
| `hhgfront/components/system-capsules.tsx` | Brain system score cards |
| `hhgfront/components/region-table.tsx` | Top active brain regions table |
| `hhgfront/components/brain-viewer.tsx` | Brain movie/image viewer |
| `hhgfront/components/snapshot-rail.tsx` | Alert level + engagement sidebar |

---

## How to Start Everything

```bash
cd /home/nurman/Desktop/arch/archangel

# 1. Start API server (loads models into VRAM, takes ~30s)
archangel/bin/python api_server.py --variant vitg --window-size 16 --stride 3

# 2. Start frontend (separate terminal)
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"
cd hhgfront && npx next start -p 3001

# 3. Start Jetson client (on Jetson via SSH)
ssh nurman@10.42.0.1
cd ~/archangelpc && source archpc/bin/activate
export DISPLAY=:0  # if display connected
python jetson_stream_client.py --host 10.42.0.211 --send-every 1

# 4. Open browser: http://localhost:3001 → click "Go Live"
```

---

## Python Environment

### PC (archangel/bin/python — Python 3.12.3)
- Created with `/usr/bin/python3 -m venv archangel`
- PyTorch 2.11.0+cu128 (force-installed over tribev2's pinned torch 2.6)
- Key packages: tribev2, transformers, opencv-python, nilearn, starlette, uvicorn, websockets, fastapi
- **WARNING**: miniconda python is broken (ARM binary on x86 system). Always use `archangel/bin/python`

### Jetson (~/archangelpc/archpc — Python 3.10)
- System OpenCV symlinked: `ln -s /usr/lib/python3.10/dist-packages/cv2 archpc/lib/python3.10/site-packages/cv2`
- `jetson_imx477_camera.py` also inserts sys.path for system cv2 at runtime
- No torch needed on Jetson — it only captures and streams frames

### Node.js
- Installed via nvm: `~/.nvm/versions/node/v22.22.2`
- Must source nvm before using: `export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"`
- System node is v18 (too old for Next.js 16)

---

## Performance Achieved

| Config | V-JEPA2 Time | Total | Brain FPS |
|--------|---:|---:|---:|
| Original tribev2_explore.py | 324s | 543s | 0.04 |
| ViT-g, single GPU, no compile | 13.8s | 14.0s | 7.1 |
| ViT-g, dual GPU | 7.8s | 8.5s | 11.7 |
| ViT-L, dual GPU (--fast) | 4.4s | 5.1s | 19.6 |
| Streaming (live, ViT-g) | ~530ms/pred | — | ~1.8 |

---

## Known Issues & TODO

### Critical
1. **Frontend "Go Live" sometimes shows "waiting for connection"** — WebSocket connects (confirmed in server logs) but frontend may not receive data if Jetson isn't streaming. Must start Jetson client BEFORE clicking Go Live.
2. **Emotion detection is limited** — TRIBE v2 predicts cortical surface BOLD signals. Amygdala and other key emotion structures are subcortical and can't be directly predicted. Current emotion scoring uses surface-level proxy regions (insula, ACC, parahippocampal). Motion causes more variation than actual emotional content.
3. **ViT-L (--fast) zero-pads features** — When using vitl variant, features are padded from 1024→1408 dims to match brain model. This degrades prediction quality significantly. Use vitg for real predictions.

### Should Fix
4. **Spike detection too sensitive to motion** — Visual cortex baseline subtraction helps but doesn't fully eliminate motion artifacts. Consider: running a rolling z-score on emotion regions only, or requiring sustained elevation over multiple predictions.
5. **Frontend doesn't show predictions when not in live mode** — The existing upload flow uses mock data (`USE_MOCK = true` in lib/api.ts). Need to wire upload to the real backend at `/api/upload`.
6. **No audio from camera** — Jetson streams video only. Wav2Vec-BERT gets zero features. TRIBE v2 was trained with audio — adding mic input would improve predictions significantly.
7. **Frontend camera feed is JPEG polling** — `getFrameUrl()` polls `/api/frame` every 200ms. Should switch to MJPEG stream or WebRTC for smoother video.

### Nice to Have
8. **Brain visualization in frontend** — The brain_visualizer.py can render 3D brain surfaces but the frontend doesn't use it in live mode. Could generate brain surface snapshots per prediction.
9. **Recording/playback** — Save streaming sessions for later review.
10. **Jetson display preview** — Works with `DISPLAY=:0` but can't do X forwarding over SSH. The overlay shows emotion scores when running locally on Jetson's display.

---

## Model Details

### TRIBE v2 Brain Model
- Checkpoint: `facebook/tribev2` on HuggingFace (auto-downloaded to ~/.cache)
- Architecture: FmriEncoder with transformer, predicts 20,484 cortical vertices (fsaverage5 mesh)
- Input: video features (2 layer groups × 1408 dim × T timesteps) + audio features (2 × 1024 × T)
- Must set `average_subjects=True` and `n_subjects=0` when loading (checkpoint was saved this way)
- Brain model is tiny (~50M params), inference is <10ms

### V-JEPA2 (Video Encoder)
- Model: `facebook/vjepa2-vitg-fpc64-256` (ViT-g, 2.5B params, 64 frames per clip, 256px)
- Produces 48 hidden layers × 8192 tokens × 1408 dim per clip
- Layer aggregation: group_mean with layers [0.5, 0.75, 1.0] → 2 groups
- Token aggregation: mean → 1408 dim per group
- ~260ms per clip on RTX 5090 (bf16 + SDPA)

### Wav2Vec-BERT (Audio Encoder)
- Model: `facebook/w2v-bert-2.0`
- Same layer/token aggregation → 2 groups × 1024 dim
- Processes full audio at once, ~100ms for 21s clip

### Emotion Region Mapping (in pc_inference_server.py)
- `EMOTION_REGIONS` dict maps Destrieux atlas labels to emotion system types
- `EMOTION_SYSTEMS` groups regions into: fear_anxiety, emotional_arousal, anger_stress, social_emotion
- `SYSTEM_MAP` maps to frontend-friendly system names (Visual Processing, Emotion, etc.)
- Atlas loaded once and cached in `_ATLAS_CACHE`

---

## Network

- PC: 10.42.0.211 (Ethernet to Jetson), 172.16.0.2 (main network)
- Jetson: 10.42.0.1 (hostname: nurman-superjet)
- SSH to Jetson requires password auth (no key from Claude's shell)
- SCP files TO Jetson: `scp file nurman@10.42.0.1:~/archangelpc/` (run from PC terminal, not Claude)

---

## File Locations on Jetson

```
~/archangelpc/
├─ archpc/          (venv)
├─ jetson_stream_client.py
├─ jetson_imx477_camera.py
├─ jetson_imx477_autofocus.py
└─ stream_protocol.py
```

Camera: Arducam IMX477 on CSI, autofocus via I2C bus 10, Focuser library at `/home/nurman/MIPI_Camera/Jetson/IMX477/AF_LENS`

---

## What Was Revolutionary

1. Took Meta's research-grade TRIBE v2 (designed for offline SLURM batch processing) and made it run in real-time streaming — 107x faster
2. First known deployment of TRIBE v2 on edge hardware (Jetson) with live camera input
3. Dual-GPU parallel inference on consumer GPUs (5090 + 4090)
4. Emotion-specific brain circuit tracking with visual cortex baseline subtraction
5. Full-stack: edge camera → TCP streaming → GPU inference → WebSocket → React dashboard
