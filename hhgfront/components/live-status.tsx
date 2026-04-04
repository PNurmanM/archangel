"use client";

import { motion, AnimatePresence } from "framer-motion";
import { BrainPrediction } from "@/lib/types";

interface LiveStatusProps {
  prediction: BrainPrediction | null;
  isLive: boolean;
  liveStatus: string;
}

export function LiveStatus({ prediction, isLive, liveStatus }: LiveStatusProps) {
  if (!isLive) return null;

  // Extract live-specific fields from the prediction (passed through from WS)
  const raw = prediction as any;
  const spike = raw?.spike ?? "steady";
  const spikePct = raw?.spike_pct ?? 0;
  const emotions = raw?.emotions ?? {};
  const timingMs = raw?.timing_ms ?? 0;
  const inferenceCount = raw?.inference_count ?? 0;
  const historyLength = raw?.history_length ?? 0;

  const fear = emotions.fear_anxiety ?? 0;
  const anger = emotions.anger_stress ?? 0;
  const arousal = emotions.emotional_arousal ?? 0;
  const social = emotions.social_emotion ?? 0;

  const isSpike = spike === "SPIKE!" || spike === "DROP!";
  const isRising = spike === "rising" || spike === "falling";

  const spikeColor = spike === "SPIKE!" ? "#ef4444" : spike === "DROP!" ? "#3b82f6" :
    isRising ? "#f59e0b" : "#22c55e";

  const stateLabel = spike === "SPIKE!" ? "EMOTIONAL SPIKE" : spike === "DROP!" ? "CALMING" :
    spike === "rising" ? "RISING" : spike === "falling" ? "FALLING" : "STEADY";

  return (
    <div className="card-surface px-5 py-4 mb-4">
      {/* Connection status */}
      {liveStatus !== "connected" && !prediction && (
        <div className="flex items-center gap-2 text-[13px] text-[rgba(245,247,250,0.48)]">
          <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
          {liveStatus === "buffering" ? "Buffering frames from Jetson..." :
           liveStatus === "connecting" ? "Connecting to inference server..." : "Waiting for connection..."}
        </div>
      )}

      {prediction && (
        <div className="space-y-3">
          {/* Top row: State + Spike indicator */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-[11px] text-[rgba(245,247,250,0.5)] font-medium tracking-wider uppercase">
                  Live #{inferenceCount} | T={historyLength} | {timingMs}ms
                </span>
              </div>
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={spike}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="flex items-center gap-2 px-3 py-1 rounded-full"
                style={{ background: `${spikeColor}20`, border: `1px solid ${spikeColor}40` }}
              >
                {isSpike && (
                  <motion.div
                    animate={{ scale: [1, 1.3, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity }}
                    className="w-2 h-2 rounded-full"
                    style={{ background: spikeColor }}
                  />
                )}
                <span className="text-[12px] font-bold" style={{ color: spikeColor }}>
                  {stateLabel} {spikePct !== 0 ? `(${spikePct > 0 ? "+" : ""}${spikePct.toFixed(0)}%)` : ""}
                </span>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Emotion bars */}
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: "Fear", value: fear, color: "#ef4444" },
              { label: "Anger", value: anger, color: "#f97316" },
              { label: "Arousal", value: arousal, color: "#eab308" },
              { label: "Social", value: social, color: "#8b5cf6" },
            ].map(({ label, value, color }) => {
              // Scale: typical range 0.01-0.06, max around 0.15
              const pct = Math.min((value / 0.08) * 100, 100);
              return (
                <div key={label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11px] text-[rgba(245,247,250,0.5)]">{label}</span>
                    <span className="text-[11px] font-mono" style={{ color }}>{value.toFixed(3)}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: color }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.3, ease: "easeOut" }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
