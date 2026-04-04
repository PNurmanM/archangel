/**
 * Generate a brain-activation video (WebM blob) from per-second frame analysis.
 * Renders a top-down brain heatmap on a canvas and records it with MediaRecorder.
 * Each second shows the activation pattern for that second of the input video.
 */

import type { FrameFeatures } from "./video-analyzer";

const SYSTEMS = [
  "Visual Processing",
  "Object/Face Recognition",
  "Attention",
  "Social Cognition",
  "Motor System",
  "Higher Cognition",
  "Touch & Sensation",
  "Memory",
  "Emotion",
];

/* Brain region hotspot positions (fraction of canvas) */
const SPOTS: { sys: string; x: number; y: number; r: number }[] = [
  { sys: "Visual Processing",       x: 0.50, y: 0.82, r: 0.13 },
  { sys: "Object/Face Recognition", x: 0.22, y: 0.62, r: 0.10 },
  { sys: "Memory",                  x: 0.78, y: 0.62, r: 0.09 },
  { sys: "Attention",               x: 0.72, y: 0.40, r: 0.11 },
  { sys: "Touch & Sensation",       x: 0.28, y: 0.40, r: 0.09 },
  { sys: "Motor System",            x: 0.34, y: 0.25, r: 0.10 },
  { sys: "Higher Cognition",        x: 0.66, y: 0.25, r: 0.10 },
  { sys: "Social Cognition",        x: 0.50, y: 0.15, r: 0.10 },
  { sys: "Emotion",                 x: 0.50, y: 0.52, r: 0.08 },
];

function mapFrameToSystems(f: FrameFeatures): Record<string, number> {
  const { brightness, complexity, motion, brightnessDelta } = f;
  const shock = brightnessDelta > 0.15 || motion > 0.55 ? 1 : 0;
  return {
    "Visual Processing":       0.04 + brightness * 0.12 + complexity * 0.10,
    "Object/Face Recognition": 0.02 + complexity * 0.14 * (1 - motion * 0.4),
    "Attention":               0.02 + motion * 0.14 + shock * 0.12 + complexity * 0.04,
    "Social Cognition":        Math.max(0, -0.01 + brightness * 0.04 + complexity * 0.03 - motion * 0.04),
    "Motor System":            0.01 + motion * 0.14,
    "Higher Cognition":        0.01 + complexity * 0.08 * (1 - motion * 0.3),
    "Touch & Sensation":       0.005 + motion * 0.05,
    "Memory":                  0.01 + brightness * 0.03 + complexity * 0.02,
    "Emotion":                 0.005 + shock * 0.20 + motion * 0.10 + brightnessDelta * 0.30,
  };
}

/* Heat-color: low → dark slate → burgundy → rose → bright pink */
function heatColor(val: number): string {
  const t = Math.min(1, Math.max(0, val * 5)); // scale 0–0.2 → 0–1
  const r = Math.round(21 + t * 196);  // 15 → 211
  const g = Math.round(24 + t * 104);  // 18 → 128
  const b = Math.round(32 + t * 142);  // 20 → 174
  return `rgb(${r},${g},${b})`;
}

function renderFrame(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  systems: Record<string, number>,
  second: number,
  duration: number,
) {
  /* Background */
  ctx.fillStyle = "#12141A";
  ctx.fillRect(0, 0, w, h);

  /* Brain outline */
  const cx = w * 0.5, cy = h * 0.50;
  const rx = w * 0.36, ry = h * 0.40;

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.clip();

  /* Base brain fill */
  ctx.fillStyle = "#1A1E26";
  ctx.fill();

  /* Region heatmap blobs */
  for (const spot of SPOTS) {
    const val = Math.abs(systems[spot.sys] ?? 0);
    if (val < 0.005) continue;
    const sx = w * spot.x;
    const sy = h * spot.y;
    const sr = Math.min(w, h) * spot.r * (1 + val * 3);

    const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr);
    const color = heatColor(val);
    grad.addColorStop(0, color);
    grad.addColorStop(0.5, color.replace("rgb", "rgba").replace(")", ",0.4)"));
    grad.addColorStop(1, "rgba(18,20,26,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(sx - sr, sy - sr, sr * 2, sr * 2);
  }

  ctx.restore();

  /* Brain outline stroke */
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(245,247,250,0.12)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  /* Central fissure */
  ctx.beginPath();
  ctx.moveTo(cx, cy - ry + 8);
  ctx.lineTo(cx, cy + ry - 8);
  ctx.strokeStyle = "rgba(245,247,250,0.08)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  ctx.stroke();
  ctx.setLineDash([]);

  /* Lateral sulci */
  ctx.beginPath();
  ctx.moveTo(cx - rx + 20, cy + 10);
  ctx.quadraticCurveTo(cx - rx * 0.3, cy - 10, cx, cy + 5);
  ctx.strokeStyle = "rgba(245,247,250,0.06)";
  ctx.lineWidth = 0.8;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx + rx - 20, cy + 10);
  ctx.quadraticCurveTo(cx + rx * 0.3, cy - 10, cx, cy + 5);
  ctx.stroke();

  /* Region labels */
  ctx.font = "bold 9px system-ui, sans-serif";
  ctx.textAlign = "center";
  for (const spot of SPOTS) {
    const val = Math.abs(systems[spot.sys] ?? 0);
    if (val < 0.01) continue;
    const alpha = Math.min(0.7, 0.15 + val * 4);
    ctx.fillStyle = `rgba(245,247,250,${alpha})`;
    ctx.fillText(
      spot.sys.length > 14 ? spot.sys.slice(0, 12) + "…" : spot.sys,
      w * spot.x,
      h * spot.y + Math.min(w, h) * spot.r + 14,
    );
  }

  /* Color bar */
  const barX = w - 22, barY = h * 0.25, barH = h * 0.5, barW = 8;
  for (let i = 0; i < barH; i++) {
    const t = 1 - i / barH;
    ctx.fillStyle = heatColor(t * 0.2);
    ctx.fillRect(barX, barY + i, barW, 1);
  }
  ctx.strokeStyle = "rgba(245,247,250,0.1)";
  ctx.lineWidth = 0.5;
  ctx.strokeRect(barX, barY, barW, barH);
  ctx.font = "8px system-ui, sans-serif";
  ctx.fillStyle = "rgba(245,247,250,0.3)";
  ctx.textAlign = "left";
  ctx.fillText("High", barX - 4, barY - 4);
  ctx.fillText("Low", barX - 2, barY + barH + 10);

  /* Timestamp */
  ctx.font = "bold 11px monospace";
  ctx.fillStyle = "rgba(245,247,250,0.35)";
  ctx.textAlign = "left";
  ctx.fillText(`${second.toFixed(0)}s / ${Math.round(duration)}s`, 10, h - 10);

  /* Title */
  ctx.font = "bold 10px system-ui, sans-serif";
  ctx.fillStyle = "rgba(245,247,250,0.25)";
  ctx.textAlign = "left";
  ctx.fillText("PREDICTED BRAIN ACTIVATION", 10, 16);
}

/**
 * Generate a brain-movie WebM blob URL.
 * Records the canvas in real-time at ~4 fps, one content-frame per second.
 */
export async function generateBrainMovie(
  frames: FrameFeatures[],
  duration: number,
  onProgress?: (pct: number) => void,
): Promise<string> {
  const W = 480, H = 360;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;

  /* Pick a supported mime type */
  let mime = "video/webm;codecs=vp8";
  if (typeof MediaRecorder !== "undefined" && !MediaRecorder.isTypeSupported(mime)) {
    mime = "video/webm";
  }

  const stream = canvas.captureStream(30); // capture at 30 fps for smooth playback
  const recorder = new MediaRecorder(stream, { mimeType: mime });
  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };

  return new Promise<string>((resolve) => {
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      resolve(URL.createObjectURL(blob));
    };

    recorder.start(100);

    const startTime = performance.now();
    const totalMs = duration * 1000;

    const tick = () => {
      const elapsed = performance.now() - startTime;
      if (elapsed >= totalMs) {
        /* Final frame */
        const last = frames[frames.length - 1];
        renderFrame(ctx, W, H, mapFrameToSystems(last), Math.round(duration), duration);
        setTimeout(() => recorder.stop(), 150);
        onProgress?.(1);
        return;
      }

      const sec = Math.floor(elapsed / 1000);
      const f = frames[Math.min(sec, frames.length - 1)];
      renderFrame(ctx, W, H, mapFrameToSystems(f), sec, duration);
      onProgress?.(elapsed / totalMs);
      requestAnimationFrame(tick);
    };

    tick();
  });
}
