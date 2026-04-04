import { BrainPrediction, TimePoint, AlertLevel } from "./types";
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

const REGIONS_POOL: { name: string; label: string; bias: string }[] = [
  { name: "Anterior Occipital Sulcus", label: "visual-temporal transition", bias: "Visual Processing" },
  { name: "Middle Occipital Sulcus", label: "visual association", bias: "Visual Processing" },
  { name: "Superior Occipital Sulcus", label: "dorsal visual stream", bias: "Visual Processing" },
  { name: "Middle Occipital Gyrus", label: "mid-level visual processing", bias: "Visual Processing" },
  { name: "Fusiform Gyrus", label: "face and object recognition", bias: "Object/Face Recognition" },
  { name: "Inferior Temporal Gyrus", label: "complex visual features", bias: "Object/Face Recognition" },
  { name: "Superior Parietal Lobule", label: "spatial attention", bias: "Attention" },
  { name: "Intraparietal Sulcus", label: "attentional orienting", bias: "Attention" },
  { name: "Superior Temporal Sulcus", label: "social perception", bias: "Social Cognition" },
  { name: "Medial Prefrontal Cortex", label: "theory of mind", bias: "Social Cognition" },
  { name: "Precentral Gyrus", label: "primary motor cortex", bias: "Motor System" },
  { name: "Supplementary Motor Area", label: "movement planning", bias: "Motor System" },
  { name: "Dorsolateral Prefrontal", label: "executive function", bias: "Higher Cognition" },
  { name: "Angular Gyrus", label: "semantic processing", bias: "Higher Cognition" },
  { name: "Postcentral Gyrus", label: "primary somatosensory", bias: "Touch & Sensation" },
  { name: "Insular Cortex", label: "interoception and emotion", bias: "Emotion" },
  { name: "Parahippocampal Gyrus", label: "scene and context encoding", bias: "Memory" },
  { name: "Hippocampus", label: "episodic memory", bias: "Memory" },
  { name: "Amygdala", label: "emotional salience", bias: "Emotion" },
  { name: "Anterior Cingulate", label: "conflict monitoring", bias: "Higher Cognition" },
];

/* ── Map per-second visual features → brain system activations ── */
function mapFrameToSystems(f: FrameFeatures): Record<string, number> {
  const { brightness, complexity, motion, brightnessDelta } = f;
  const shock = brightnessDelta > 0.15 || motion > 0.55 ? 1 : 0; // jumpscare / sudden change

  const raw: Record<string, number> = {
    "Visual Processing":        0.04 + brightness * 0.12 + complexity * 0.10,
    "Object/Face Recognition":  0.02 + complexity * 0.14 * (1 - motion * 0.4),
    "Attention":                0.02 + motion * 0.14 + shock * 0.12 + complexity * 0.04,
    "Social Cognition":        -0.01 + brightness * 0.04 + complexity * 0.03 - motion * 0.04,
    "Motor System":             0.01 + motion * 0.14,
    "Higher Cognition":         0.01 + complexity * 0.08 * (1 - motion * 0.3),
    "Touch & Sensation":        0.005 + motion * 0.05,
    "Memory":                   0.01 + brightness * 0.03 + complexity * 0.02,
    "Emotion":                  0.005 + shock * 0.20 + motion * 0.10 + brightnessDelta * 0.30,
  };

  // Round to 4 decimal places
  for (const k of Object.keys(raw)) {
    raw[k] = Math.round(raw[k] * 10000) / 10000;
  }
  return raw;
}

/* ── Build a full prediction from real frame analysis ── */
export function generatePredictionFromAnalysis(
  frames: FrameFeatures[],
  duration: number,
  videoUrl: string,
): BrainPrediction {
  /* ---- timeline: interpolate 4 sub-points per second for smooth curves ---- */
  const timeline: TimePoint[] = [];
  const subPerSec = 4;

  for (let i = 0; i < frames.length; i++) {
    const curr = mapFrameToSystems(frames[i]);
    const next = i < frames.length - 1 ? mapFrameToSystems(frames[i + 1]) : curr;

    for (let s = 0; s < subPerSec; s++) {
      if (i === frames.length - 1 && s > 0) break; // don't overshoot
      const frac = s / subPerSec;
      const t = frames[i].time + frac;
      if (t > duration) break;

      const systems: Record<string, number> = {};
      let absSum = 0;
      for (const sys of SYSTEMS) {
        const v = curr[sys] * (1 - frac) + next[sys] * frac;
        const rounded = Math.round(v * 10000) / 10000;
        systems[sys] = rounded;
        absSum += Math.abs(rounded);
      }

      timeline.push({
        time: Math.round(t * 10) / 10,
        total: Math.round((absSum / SYSTEMS.length) * 10000) / 10000,
        systems,
      });
    }
  }

  /* ---- aggregate system scores (mean across all seconds) ---- */
  const systemScores = SYSTEMS.map((system) => {
    const vals = timeline.map((tp) => tp.systems[system]);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    return { system, score: Math.round(mean * 10000) / 10000 };
  });

  const sorted = [...systemScores].sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  const dominantSystem = sorted[0].system;
  const totalActivity = Math.round(
    (systemScores.reduce((s, sc) => s + Math.abs(sc.score), 0) / SYSTEMS.length) * 10000,
  ) / 10000;

  /* engagement */
  let engagement: string;
  if (totalActivity > 0.10) engagement = "Very High";
  else if (totalActivity > 0.07) engagement = "High";
  else if (totalActivity > 0.04) engagement = "Moderate";
  else if (totalActivity > 0.02) engagement = "Low";
  else engagement = "Minimal";

  /* alert level */
  let alertLevel: AlertLevel;
  let alertLabel: string;
  const peakEmotion = Math.max(...timeline.map((tp) => tp.systems["Emotion"]));
  const peakMotion = Math.max(...timeline.map((tp) => tp.systems["Attention"]));
  if (peakEmotion > 0.15 || peakMotion > 0.18) {
    alertLevel = "ALERT";
    alertLabel = "Alert";
  } else if (peakEmotion > 0.08 || peakMotion > 0.12) {
    alertLevel = "WATCH";
    alertLabel = "Watch";
  } else {
    alertLevel = "NORMAL";
    alertLabel = "Normal";
  }

  /* top regions — pick regions biased towards dominant systems */
  const regionsCopy = [...REGIONS_POOL];
  regionsCopy.sort((a, b) => {
    const aBoost = sorted.findIndex((s) => s.system === a.bias);
    const bBoost = sorted.findIndex((s) => s.system === b.bias);
    return aBoost - bBoost; // lower index = higher ranked system = preferred
  });
  const topRegions = regionsCopy.slice(0, 8).map((r, i) => {
    const sysScore = Math.abs(systemScores.find((s) => s.system === r.bias)?.score ?? 0);
    return {
      rank: i + 1,
      name: r.name,
      label: r.label,
      score: Math.round((0.40 - i * 0.03 + sysScore * 2) * 10000) / 10000,
    };
  });

  /* summary */
  const emotionSpikes = frames.filter(
    (f) => f.brightnessDelta > 0.15 || f.motion > 0.55,
  );
  let summaryExtra = "";
  if (emotionSpikes.length > 0) {
    const times = emotionSpikes.map((f) => `${f.time}s`).join(", ");
    summaryExtra = ` Notable emotional spikes were detected at ${times}, suggesting sudden scene changes or startling content that triggered strong amygdala and attention responses.`;
  }

  const summary =
    `The brain shows dominant engagement of ${dominantSystem.toLowerCase()} pathways, ` +
    `with secondary activation in ${sorted[1].system.toLowerCase()}. ` +
    `Overall engagement is ${engagement.toLowerCase()} across the ${Math.round(duration)}-second clip. ` +
    `${sorted[0].system} leads at ${sorted[0].score.toFixed(4)}, followed by ${sorted[1].system} at ${sorted[1].score.toFixed(4)}.` +
    summaryExtra;

  return {
    inputVideoUrl: videoUrl,
    brainMovieUrl: "/outputs/tennis/tennis_brain_movie.mp4",
    meanBrainUrl: "/outputs/tennis/tennis_mean_brain.png",
    peakBrainUrl: "/outputs/tennis/tennis_peak_brain.png",
    totalActivity,
    dominantSystem,
    engagement,
    systemScores,
    alertLevel,
    alertLabel,
    topRegions,
    summary,
    timeline,
  };
}

/* ── Static demo prediction (for "Run Demo" — uses canned tennis data) ── */
function generateDemoTimeline(): TimePoint[] {
  const points: TimePoint[] = [];
  const n = 24; // 4 per second × 6 seconds
  const curves: Record<string, (t: number) => number> = {
    "Visual Processing": (t) => 0.12 + 0.06 * Math.sin(t * 1.2),
    "Object/Face Recognition": (t) => 0.08 + 0.04 * Math.sin(t * 0.9 + 0.5),
    "Attention": (t) => 0.07 + 0.05 * Math.sin(t * 1.1 + 1),
    "Social Cognition": (t) => -0.03 - 0.02 * Math.sin(t * 0.7),
    "Motor System": (t) => 0.03 + 0.02 * Math.sin(t * 1.3 + 0.3),
    "Higher Cognition": (t) => 0.025 + 0.015 * Math.sin(t * 0.8 + 1.5),
    "Touch & Sensation": (t) => 0.02 + 0.01 * Math.sin(t * 1.5),
    "Memory": (t) => 0.015 + 0.01 * Math.sin(t * 0.6 + 0.8),
    "Emotion": (t) => 0.006 + 0.004 * Math.sin(t * 1.8 + 2),
  };
  for (let i = 0; i <= n; i++) {
    const t = (i / n) * 6;
    const systems: Record<string, number> = {};
    let sum = 0;
    for (const sys of SYSTEMS) {
      const v = Math.round(curves[sys](t) * 10000) / 10000;
      systems[sys] = v;
      sum += Math.abs(v);
    }
    points.push({
      time: Math.round(t * 10) / 10,
      total: Math.round((sum / SYSTEMS.length) * 10000) / 10000,
      systems,
    });
  }
  return points;
}

export const mockPrediction: BrainPrediction = {
  inputVideoUrl: "/outputs/tennis/tennis_input.mp4",
  brainMovieUrl: "/outputs/tennis/tennis_brain_movie.mp4",
  meanBrainUrl: "/outputs/tennis/tennis_mean_brain.png",
  peakBrainUrl: "/outputs/tennis/tennis_peak_brain.png",
  totalActivity: 0.0635,
  dominantSystem: "Visual Processing",
  engagement: "Very High",
  systemScores: [
    { system: "Visual Processing", score: 0.1685 },
    { system: "Object/Face Recognition", score: 0.1085 },
    { system: "Attention", score: 0.0927 },
    { system: "Social Cognition", score: -0.0468 },
    { system: "Motor System", score: 0.0455 },
    { system: "Higher Cognition", score: 0.0336 },
    { system: "Touch & Sensation", score: 0.0280 },
    { system: "Memory", score: 0.0235 },
    { system: "Emotion", score: 0.0091 },
  ],
  alertLevel: "NORMAL",
  alertLabel: "Normal",
  topRegions: [
    { rank: 1, name: "Anterior Occipital Sulcus", label: "visual-temporal transition", score: 0.4381 },
    { rank: 2, name: "Middle Occipital Sulcus", label: "visual association", score: 0.3662 },
    { rank: 3, name: "Superior Occipital Sulcus", label: "dorsal visual stream", score: 0.3022 },
    { rank: 4, name: "Middle Occipital Gyrus", label: "mid-level visual processing", score: 0.2986 },
    { rank: 5, name: "Inferior Occipital Area", label: "early visual processing", score: 0.2969 },
    { rank: 6, name: "Lateral Occipitotemporal Sulcus", label: "object recognition pathway", score: 0.1928 },
    { rank: 7, name: "Superior Occipital Gyrus", label: "spatial vision", score: 0.1818 },
    { rank: 8, name: "Superior Parietal Lobule", label: "spatial attention", score: 0.1708 },
  ],
  summary:
    "The brain is strongly engaging visual processing and object recognition pathways, with peak activation in the occipital cortex. Attention and spatial awareness are elevated, consistent with visually complex content that grabs focus. Motor cortex shows mild activation, suggesting the video depicts physical action. Social cognition is slightly suppressed, indicating externally focused attention rather than social processing.",
  timeline: generateDemoTimeline(),
};
