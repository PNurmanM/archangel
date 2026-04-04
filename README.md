# HHG — Brain Activity Prediction with TRIBE v2

## What is this?

You give it a photo, it predicts how a human brain would react to it — the same kind of data you'd get from an fMRI brain scan. Then it tells you in plain English which parts of the brain lit up and what that means (faces detected, emotions triggered, attention grabbed, etc).

The end goal is to hook this up to a **Jetson Orin Nano + camera** for real-time brain response prediction.

---

## What is TRIBE v2?

An AI model by Meta trained on **1,000+ hours of real fMRI brain scans** from **720 people**. It learned how brains respond to what we see, hear, and read. Now it can predict what a brain scan would look like for any new image — no human in an MRI machine needed.

### The 3 models inside it

| Model | Handles | Made by |
|-------|---------|---------|
| **V-JEPA2** | Images/video | Meta |
| **Wav2Vec-BERT** | Audio/speech | Google |
| **LLaMA 3.2** | Text/language | Meta |

These feed into a Transformer that maps their outputs onto a 3D brain surface.

### Where does it run?

Locally on your machine using **PyTorch**. There is no cloud API. The model weights (~5GB) are downloaded from HuggingFace on first run and cached locally.

```
Your script
  -> tribev2 library (Meta's Python package)
    -> HuggingFace Transformers (loads the 3 sub-models)
      -> PyTorch (runs all the math)
        -> CUDA (sends math to your NVIDIA GPU)
```

---

## What does the output look like?

### Raw data

A numpy array shaped `(n_timesteps, 20484)`:
- **n_timesteps** = seconds of stimulus (a 2-second image gives ~2 rows)
- **20,484** = vertices on a 3D brain surface model called **fsaverage5**
- Each number = predicted fMRI activation at that brain location

### Brain summary

The script maps those raw vertex numbers to named brain regions and tells you what's happening:

```
BRAIN ACTIVITY SUMMARY
============================================================

  TOP 15 MOST ACTIVE BRAIN REGIONS:
  #    Region                                      Score  What it does
  1    Inferior Occipital Area                  +0.1242  Early visual processing
  2    Fusiform Gyrus                           +0.0765  Face and object recognition
  3    Superior Temporal Sulcus                 +0.0456  Social perception, theory of mind
  ...

  ACTIVITY BY BRAIN SYSTEM:
    Social Cognition                    +0.0456  [    ACTIVE]  #############
    Object/Face Recognition             +0.0402  [    ACTIVE]  ############
    Language & Communication            +0.0191  [    ACTIVE]  #####
    ...

  EMOTIONAL & COGNITIVE PROFILE:
    [    STRONG]  Face/Body Detection: Fusiform gyrus active — recognizing faces
    [    ACTIVE]  Social Processing: Image involves people or social cues
    [      MILD]  Empathy: Brain is relating to others' emotions

  WHAT YOUR BRAIN IS DOING WITH THIS IMAGE:
    The brain is engaging reading social cues, empathy, understanding others.
      Strongest contributor: Superior Temporal Sulcus
      This means: Reading social cues, understanding others' intentions

  OVERALL BRAIN ENGAGEMENT:
    MODERATE — This image produces a noticeable brain response.
```

### How the summary is generated

This is important to understand — the summary is **not AI-generated**. It works like this:

1. **TRIBE v2** outputs raw numbers (20,484 floats). This is the science part.
2. **Nilearn's Destrieux atlas** maps vertex numbers to region names (e.g., vertex 5281 -> "Fusiform Gyrus"). Also science.
3. **A hardcoded lookup table** in the script maps region names to descriptions (e.g., "Fusiform Gyrus" -> "face recognition"). Written by hand based on neuroscience textbooks.
4. **Hardcoded thresholds** decide what counts as "active" (e.g., score > 0.01 = active). These are arbitrary, not scientifically calibrated.

Negative scores mean "below baseline" — the brain is prioritizing other regions. It does NOT mean something bad or aversive.

---

## Project structure

```
hhg/
├── tribev2_explore.py   # Main script — image in, brain summary out
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── cache/               # Model weights (created on first run)
└── *_brain_preds.npy    # Saved raw predictions per image
```

### Dependencies

| Package | Why |
|---------|-----|
| **tribev2** | Meta's model — the core brain prediction engine |
| **opencv-python** | Opens images + converts them to video (TRIBE v2 only takes video) |
| **numpy** | Handles the 20,484-number arrays |
| **nilearn** | Maps vertex numbers to named brain regions using the Destrieux atlas |
| **huggingface_hub** | Downloads model weights from HuggingFace |

---

## Setup (Windows 11)

### 1. Create and activate virtual environment

```powershell
python -m venv hhg
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
hhg\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install nilearn huggingface_hub
```

### 3. Install PyTorch with CUDA (if you have an NVIDIA GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Login to HuggingFace

The model weights are hosted on HuggingFace. LLaMA 3.2 is gated, so you need permission.

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and **accept the license**
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **Read** token
4. Copy the token (starts with `hf_...`)
5. Run:

```bash
python -c "from huggingface_hub import login; login()"
```

Paste token when prompted (won't show characters — normal). Press Enter.

### 5. Run

```bash
python tribev2_explore.py photo.jpg
```

With 3D brain visualization:

```bash
python tribev2_explore.py photo.jpg --visualize
```

---

## How the script works (step by step)

1. **You provide an image** (jpg, png, etc)
2. **OpenCV converts it to a 2-second video** — TRIBE v2 only accepts video input
3. **TRIBE v2 processes the video** through V-JEPA2 and maps features onto the brain surface
4. **Raw output**: 20,484 activation values per timestep
5. **Nilearn atlas** maps vertices to named brain regions
6. **Hardcoded lookup table** translates region names to plain English
7. **Summary printed** — which brain systems are active, emotional profile, what it all means
8. **Raw data saved** as `.npy` file for further analysis

---

## Windows compatibility patches

TRIBE v2 was built for Linux. We had to patch two bugs in Meta's code:

1. **`demo_utils.py` line 201** — Windows `Path` uses backslashes which breaks HuggingFace repo IDs. Fixed with `.replace("\\", "/")`.
2. **`demo_utils.py` line 205** — Config file contains serialized `PosixPath` objects that can't load on Windows. Fixed by adding a custom YAML constructor that converts them to `Path`.

These patches are in `hhg/Lib/site-packages/tribev2/demo_utils.py`. If you reinstall tribev2, you'll need to reapply them.

---

## What's next

- [x] Get TRIBE v2 running locally
- [x] Test with sample images
- [x] Add human-readable brain region summaries
- [ ] Deploy on Jetson Orin Nano
- [ ] Hook up camera to capture photos automatically
- [ ] Run predictions in real time on captured photos

---

## Links

- [TRIBE v2 GitHub](https://github.com/facebookresearch/tribev2)
- [TRIBE v2 on HuggingFace](https://huggingface.co/facebook/tribev2)
- [Meta AI Blog Post](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/)
- [Interactive Demo](https://aidemos.atmeta.com/tribev2/)
- [Demo Notebook](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
