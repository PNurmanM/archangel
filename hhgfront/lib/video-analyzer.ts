/**
 * Client-side video frame analysis.
 * Extracts per-second visual features (brightness, motion, complexity)
 * used to drive mock brain-activity predictions.
 */

export interface FrameFeatures {
  time: number;
  /** 0-1 average luminance */
  brightness: number;
  /** 0-1 colour variance / visual complexity */
  complexity: number;
  /** 0-1 inter-frame difference (motion / scene change) */
  motion: number;
  /** absolute brightness delta from previous frame — detects flashes/jumpscares */
  brightnessDelta: number;
}

/**
 * Sample one frame per second, compute visual features.
 * Uses a tiny offscreen canvas for performance.
 */
export async function analyzeVideoFrames(
  file: File,
): Promise<{ duration: number; frames: FrameFeatures[] }> {
  const url = URL.createObjectURL(file);

  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    video.onloadeddata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video"));
  });

  const duration = Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : 6;

  const W = 160;
  const H = 90;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

  const frames: FrameFeatures[] = [];
  let prevPixels: Uint8ClampedArray | null = null;
  let prevBrightness = 0;

  const totalSeconds = Math.ceil(duration);

  for (let s = 0; s <= totalSeconds; s++) {
    const t = Math.min(s, duration - 0.01);

    // Seek to time
    video.currentTime = t;
    await new Promise<void>((resolve) => {
      video.onseeked = () => resolve();
    });

    ctx.drawImage(video, 0, 0, W, H);
    const { data: pixels } = ctx.getImageData(0, 0, W, H);
    const pixelCount = W * H;

    // Average brightness (luminance)
    let lumSum = 0;
    let rSum = 0, gSum = 0, bSum = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
      lumSum += 0.299 * r + 0.587 * g + 0.114 * b;
      rSum += r; gSum += g; bSum += b;
    }
    const brightness = lumSum / pixelCount / 255;
    const avgR = rSum / pixelCount;
    const avgG = gSum / pixelCount;
    const avgB = bSum / pixelCount;

    // Colour variance (visual complexity)
    let varSum = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      varSum +=
        Math.abs(pixels[i] - avgR) +
        Math.abs(pixels[i + 1] - avgG) +
        Math.abs(pixels[i + 2] - avgB);
    }
    const complexity = Math.min(1, varSum / pixelCount / 200);

    // Inter-frame difference (motion / scene change)
    let motion = 0;
    if (prevPixels) {
      let diffSum = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        diffSum +=
          Math.abs(pixels[i] - prevPixels[i]) +
          Math.abs(pixels[i + 1] - prevPixels[i + 1]) +
          Math.abs(pixels[i + 2] - prevPixels[i + 2]);
      }
      motion = Math.min(1, diffSum / pixelCount / 180);
    }

    const brightnessDelta = Math.abs(brightness - prevBrightness);

    frames.push({
      time: s,
      brightness,
      complexity,
      motion,
      brightnessDelta,
    });

    prevPixels = new Uint8ClampedArray(pixels);
    prevBrightness = brightness;
  }

  URL.revokeObjectURL(url);
  return { duration, frames };
}
