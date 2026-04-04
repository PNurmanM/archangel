"use client";

import { useEffect, useRef, useState, RefObject } from "react";
import { TimePoint } from "@/lib/types";

/*
  Animated SVG brain — top-down view with 9 system hotspots.
  Driven by timeline data, syncs to the input video's currentTime.
*/

interface BrainCanvasProps {
  timeline: TimePoint[];
  inputVideoRef: RefObject<HTMLVideoElement | null>;
}

/* Region positions (% of viewBox 400×400) and the system they represent */
const REGIONS: { system: string; cx: number; cy: number; r: number; label: string }[] = [
  { system: "Visual Processing",        cx: 200, cy: 340, r: 36, label: "Visual" },
  { system: "Object/Face Recognition",  cx: 100, cy: 275, r: 30, label: "Recognition" },
  { system: "Memory",                   cx: 305, cy: 275, r: 28, label: "Memory" },
  { system: "Attention",                cx: 290, cy: 180, r: 32, label: "Attention" },
  { system: "Touch & Sensation",        cx: 110, cy: 185, r: 26, label: "Sensory" },
  { system: "Motor System",             cx: 140, cy: 120, r: 28, label: "Motor" },
  { system: "Higher Cognition",         cx: 265, cy: 120, r: 28, label: "Cognition" },
  { system: "Social Cognition",         cx: 200, cy: 80,  r: 30, label: "Social" },
  { system: "Emotion",                  cx: 200, cy: 220, r: 26, label: "Emotion" },
];

/* Neural connection pairs (indices into REGIONS) */
const CONNECTIONS: [number, number][] = [
  [0, 1], [0, 2], [0, 3], [0, 4],  // visual → temporal/parietal
  [1, 4], [2, 3],                    // temporal cross
  [3, 6], [4, 5],                    // parietal → frontal
  [5, 7], [6, 7],                    // frontal → prefrontal
  [8, 1], [8, 2], [8, 7], [8, 0],   // emotion hub
  [5, 6],                            // motor ↔ cognition
];

function interpolateTimeline(timeline: TimePoint[], time: number): Record<string, number> {
  if (!timeline.length) return {};
  if (time <= timeline[0].time) return timeline[0].systems;
  if (time >= timeline[timeline.length - 1].time) return timeline[timeline.length - 1].systems;

  // Find bracketing points
  let lo = 0;
  for (let i = 1; i < timeline.length; i++) {
    if (timeline[i].time >= time) { lo = i - 1; break; }
  }
  const hi = Math.min(lo + 1, timeline.length - 1);
  const span = timeline[hi].time - timeline[lo].time;
  const frac = span > 0 ? (time - timeline[lo].time) / span : 0;

  const result: Record<string, number> = {};
  for (const key of Object.keys(timeline[lo].systems)) {
    result[key] = timeline[lo].systems[key] * (1 - frac) + timeline[hi].systems[key] * frac;
  }
  return result;
}

function activationToOpacity(value: number): number {
  const abs = Math.abs(value);
  // Map 0 → 0.06, 0.05 → 0.25, 0.15+ → 0.85
  return Math.min(0.9, 0.06 + abs * 5);
}

function activationToGlow(value: number): number {
  const abs = Math.abs(value);
  return Math.min(20, abs * 100);
}

export function BrainCanvas({ timeline, inputVideoRef }: BrainCanvasProps) {
  const [systems, setSystems] = useState<Record<string, number>>({});
  const [currentTime, setCurrentTime] = useState(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const tick = () => {
      const video = inputVideoRef.current;
      if (video && !video.paused) {
        const t = video.currentTime;
        setCurrentTime(t);
        setSystems(interpolateTimeline(timeline, t));
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    // Initial state
    setSystems(interpolateTimeline(timeline, 0));
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [timeline, inputVideoRef]);

  const duration = timeline.length ? timeline[timeline.length - 1].time : 1;

  return (
    <div className="relative w-full h-full flex items-center justify-center select-none">
      <svg viewBox="0 0 400 400" className="w-full h-full max-h-full" xmlns="http://www.w3.org/2000/svg">
        <defs>
          {/* Glow filter */}
          <filter id="brainGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Soft radial for each region */}
          <radialGradient id="regionGrad">
            <stop offset="0%" stopColor="#C75B6E" stopOpacity="1" />
            <stop offset="70%" stopColor="#C75B6E" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#C75B6E" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="regionGradNeg">
            <stop offset="0%" stopColor="#617086" stopOpacity="0.8" />
            <stop offset="70%" stopColor="#617086" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#617086" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Brain outline — top-down view */}
        <ellipse
          cx="200" cy="210" rx="155" ry="175"
          fill="none"
          stroke="rgba(245,247,250,0.08)"
          strokeWidth="1.5"
        />
        {/* Central fissure */}
        <line
          x1="200" y1="42" x2="200" y2="378"
          stroke="rgba(245,247,250,0.06)"
          strokeWidth="1"
          strokeDasharray="4 6"
        />
        {/* Lateral fissure hints */}
        <path d="M80,230 Q140,210 200,220" fill="none" stroke="rgba(245,247,250,0.05)" strokeWidth="0.8" />
        <path d="M320,230 Q260,210 200,220" fill="none" stroke="rgba(245,247,250,0.05)" strokeWidth="0.8" />

        {/* Neural connections */}
        {CONNECTIONS.map(([a, b], i) => {
          const ra = REGIONS[a], rb = REGIONS[b];
          const va = Math.abs(systems[ra.system] ?? 0);
          const vb = Math.abs(systems[rb.system] ?? 0);
          const intensity = Math.min(0.5, (va + vb) * 2);
          return (
            <line
              key={i}
              x1={ra.cx} y1={ra.cy} x2={rb.cx} y2={rb.cy}
              stroke="#C75B6E"
              strokeWidth={0.5 + intensity * 2}
              strokeOpacity={0.04 + intensity * 0.4}
              style={{ transition: "all 0.25s ease" }}
            />
          );
        })}

        {/* Region hotspots */}
        {REGIONS.map((reg) => {
          const value = systems[reg.system] ?? 0;
          const opacity = activationToOpacity(value);
          const glow = activationToGlow(value);
          const isNeg = value < 0;

          return (
            <g key={reg.system}>
              {/* Glow aura */}
              <circle
                cx={reg.cx} cy={reg.cy}
                r={reg.r + glow * 0.6}
                fill={isNeg ? "#617086" : "#C75B6E"}
                opacity={opacity * 0.15}
                style={{ transition: "all 0.25s ease" }}
              />
              {/* Main hotspot */}
              <circle
                cx={reg.cx} cy={reg.cy}
                r={reg.r}
                fill={`url(#${isNeg ? "regionGradNeg" : "regionGrad"})`}
                opacity={opacity}
                style={{ transition: "all 0.25s ease" }}
              />
              {/* Core dot */}
              <circle
                cx={reg.cx} cy={reg.cy}
                r={3 + opacity * 4}
                fill={isNeg ? "#8792A2" : "#D98090"}
                opacity={0.3 + opacity * 0.7}
                style={{ transition: "all 0.25s ease" }}
              />
              {/* Label */}
              <text
                x={reg.cx} y={reg.cy + reg.r + 14}
                textAnchor="middle"
                fontSize="9"
                fontWeight="600"
                letterSpacing="0.05em"
                fill="rgba(245,247,250,0.4)"
                style={{ transition: "fill 0.25s ease", fill: opacity > 0.4 ? "rgba(245,247,250,0.65)" : "rgba(245,247,250,0.3)" }}
              >
                {reg.label}
              </text>
            </g>
          );
        })}

        {/* Timestamp */}
        <text
          x="12" y="392"
          fontSize="10"
          fontWeight="600"
          fill="rgba(245,247,250,0.3)"
          className="tabular"
        >
          {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
        </text>
      </svg>
    </div>
  );
}
