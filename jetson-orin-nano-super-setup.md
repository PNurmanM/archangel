# Jetson Orin Nano Super — Setup & Reference Guide

compiled from nurman's setup session, april 3-4 2026

---

## Hardware

- **Board:** NVIDIA Jetson Orin Nano Super Developer Kit (8GB unified RAM)
- **Camera:** Arducam IMX477 12MP with motorized focus (I2C controlled)
- **Storage:** SD card (boot), 2TB NVMe SSD available
- **Hostname:** nurman-superjet

---

## JetPack & Flashing

### Which JetPack?

- **JetPack 6.2.x** is the correct version for Orin Nano Super (L4T r36.4.x)
- JetPack 7.x is Thor-only right now. Orin Series support comes in **JetPack 7.2 (Q2 2026)** — still in development
- JetPack 6.2.2 is the latest stable release (Jetson Linux 36.5)

### How to Flash

1. Download the SD card image from https://developer.nvidia.com/embedded/jetpack-sdk-62
2. Unzip — there's an `.img` file inside
3. Flash to microSD (or NVMe via USB adapter) with **balenaEtcher**
4. Insert into Jetson, boot

**No ISO exists** for Jetson. NVIDIA uses pre-built `.img` files, not installers.

### Firmware Warning

If the Jetson still has factory firmware, it's incompatible with JetPack 6.x. Symptoms: black screen for 3+ minutes on boot. Fix: flash JetPack 5.1.3 first to update UEFI firmware, then flash JetPack 6.x.

### Upgrading to 6.2.2 from 6.2

```bash
# edit sources to point to r36.4 repo
sudo nano /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# change version to r36.4 in both lines

sudo apt update
sudo apt dist-upgrade
sudo apt install --fix-broken -o Dpkg::Options::="--force-overwrite"
# reboot after
```

---

## Removing Snap (Complete Guide)

Ubuntu 22.04 on JetPack ships snap by default. Firefox and Chromium are snap wrappers. Here's how to nuke it.

### Step 1: Remove all snap packages

```bash
sudo snap remove --purge firefox
sudo snap remove --purge chromium
sudo snap remove --purge cups
sudo snap remove --purge snap-store
sudo snap remove --purge snapd-desktop-integration
sudo snap remove --purge gtk-common-themes
sudo snap remove --purge gnome-42-2204
sudo snap remove --purge core22
sudo snap remove --purge bare
```

Note: `cups` (printing system) holds a dependency on `core22`. Remove cups first if core22 won't uninstall.

### Step 2: Remove snapd

```bash
sudo apt remove -y --purge snapd
sudo apt autoremove -y
```

### Step 3: Block snap from ever coming back

```bash
sudo nano /etc/apt/preferences.d/nosnap.pref
```

Paste:

```
Package: snapd
Pin: release a=*
Pin-Priority: -10
```

This sets snapd's apt priority to -10 (below 0 = never install, even as a dependency). Normal packages are ~500.

### Step 4: Clean up leftover directories

```bash
rm -rf ~/snap
sudo rm -rf /snap /var/snap /var/lib/snapd
```

### Step 5: Install Firefox as a real deb

Ubuntu's default `firefox` apt package is a snap wrapper. You need the Mozilla Team PPA:

```bash
sudo add-apt-repository ppa:mozillateam/ppa
```

Then pin it to prefer the PPA version:

```bash
sudo nano /etc/apt/preferences.d/mozilla-firefox
```

Paste:

```
Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001
```

Then:

```bash
sudo apt update
sudo apt install firefox
```

### Notes

- Only Firefox and Chromium have the snap wrapper problem. Everything else installs normally via apt.
- Chromium is not needed if you have Firefox.
- `nano` is a normal deb package, not a snap.

---

## Power & Shutdown

- The Orin Nano dev kit has **no power button**. It powers on when you plug in the DC barrel jack.
- To shut down: `sudo shutdown now` or `sudo halt`
- Pulling the plug is fine if you're at GRUB/bootloader (no OS running, nothing to corrupt)

---

## RAM & Model Size Limits

- 8GB unified RAM shared between CPU and GPU
- **Hard ceiling: stay under ~6GB models** to leave room for OS + CUDA overhead
- A 16GB model will NOT run regardless of storage speed (NVMe or SD)
- NVMe is faster for loading but the bottleneck is RAM, not storage
- Swapping to disk makes inference unusably slow

### Models that fit on 8GB Orin Nano

- gemma3:4b (~3GB)
- gemma4:e2b Q4_K_M (~3GB text-only GGUF from HF, or ~7.2GB with vision via ollama)
- phi-4-mini (~2.5GB)
- deepseek-r1:1.5b
- qwen2.5:1.5b-instruct

### Models that DON'T fit

- Anything over ~6GB model weight
- gemma4:e2b Q8 (5.3GB model + 986MB vision = too tight)
- gemma4:e4b (9.6GB)
- Any 26B+ model

---

## Ollama on Jetson

### Installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Known Issues

**500 Internal Server Error / unable to load model:**

Multiple possible causes:

1. **Ollama version too old for the model** — gemma4 (released April 2026) requires a pre-release version of ollama newer than 0.19.0. Even 0.19.0 gives a 412 error. Wait for stable release or grab the RC from GitHub releases.

2. **CUDA compatibility** — the standard install script may pull a version incompatible with JetPack's CUDA. Fix: use dustynv's Jetson-specific container:
   ```bash
   sudo docker run -d --runtime nvidia --name ollama -p 11434:11434 dustynv/ollama:r36.4.0
   ```

3. **Specific version regression** — ollama 0.12.10 broke on Jetson, 0.12.9 worked:
   ```bash
   sudo systemctl stop ollama
   curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.12.9 sh
   ```

### Gemma 4 Notes

- Released April 2, 2026 — very new, support is still stabilizing
- E2B = "Effective 2B" (5.12B actual params, designed for edge devices)
- Ollama's gemma4:e2b is 7.2GB because it bundles the vision encoder (SigLIP) + language model + projector
- HuggingFace GGUFs are smaller because they're often text-only weights
- Day-one implementations across tools have known tokenizer bugs and quantization issues — give it a week

---

## Arducam IMX477 Motorized Focus Camera

### Driver Installation

Arducam cameras need kernel drivers matched to your exact L4T version. Use their auto-detect script:

```bash
cd ~
wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.3/install_full.sh
chmod +x install_full.sh
./install_full.sh -m imx477
sudo reboot
```

### Verify Camera

```bash
ls /dev/video*
# should show a video device

# test with NVIDIA's capture tool
nvgstcapture-1.0

# or gstreamer pipeline
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! xvimagesink
```

### Motorized Focus Control

The focus motor is controlled via I2C on the camera bus — no extra wiring needed.

```bash
cd ~
git clone https://github.com/ArduCAM/MIPI_Camera.git
cd MIPI_Camera/Jetson/IMX477/Motorized_Focus
```

**Manual focus (keyboard arrow keys):**
```bash
python3 focus_control.py
```

**Autofocus (OpenCV-based, sweeps focal range and locks on sharpest frame):**
```bash
python3 autofocus.py
```

**Raw I2C control (advanced):**
```bash
# find the focus motor's I2C address (usually 0x0c)
i2cdetect -y -r 10

# then set focus position directly with i2cset
```

### Camera Specs

- Sensor: Sony IMX477, 1/2.3", 12.3MP
- Max resolution: 4056 x 3040
- Frame rates (L4T 35.x+): 4032x3040@20fps, 3840x2160@30fps, 1920x1080@60fps
- FOV: 100°(D) / 87°(H) / 71°(V)
- IR sensitivity: visible light only (some models have auto IR-cut filter)

---

## NVIDIA Developer Resources Filter

When browsing NVIDIA's developer catalog and filtering by board, the Orin Nano Super isn't listed separately. Use **Jetson AGX Orin** — same Orin family, same JetPack/L4T software stack. Just be aware the Orin Nano has less compute/memory so heavier demos may not fit.

---

## Recommended Next Steps

- [ ] Move boot to NVMe SSD for better performance (SD card is slow)
- [ ] Wait for ollama stable update with gemma4 support, or use gemma3:4b in the meantime
- [ ] Set power mode to MAXN SUPER for full performance: click NVIDIA icon in top bar → Power mode → MAXN SUPER
- [ ] Upgrade to JetPack 6.2.2 via apt for Docker 28.x fix and CUDA memory allocation fix
- [ ] Test camera with `nvgstcapture-1.0` after driver install + reboot
