"use client";

import { motion } from "framer-motion";
import { useState, RefObject } from "react";

interface BrainViewerProps {
  brainMovieUrl: string;
  meanBrainUrl?: string;
  peakBrainUrl?: string;
  videoRef?: RefObject<HTMLVideoElement | null>;
}

type ViewMode = "video" | "mean" | "peak";

export function BrainViewer({ brainMovieUrl, meanBrainUrl, peakBrainUrl, videoRef }: BrainViewerProps) {
  const [hovered, setHovered] = useState(false);
  const [mode, setMode] = useState<ViewMode>("video");

  const tabs: { key: ViewMode; label: string; available: boolean }[] = [
    { key: "video", label: "Video", available: true },
    { key: "mean", label: "Mean", available: !!meanBrainUrl },
    { key: "peak", label: "Peak", available: !!peakBrainUrl },
  ];

  return (
    <div className="card-elevated flex flex-col h-full overflow-hidden brain-halo">
      <div className="flex items-center justify-between px-5 pt-4 pb-2 relative z-10">
        <span className="section-label">Neural Visualization</span>
        <div className="flex gap-0.5 bg-[#15181D] rounded-lg p-0.5 border border-[rgba(255,255,255,0.06)]">
          {tabs.filter(t => t.available).map((tab) => (
            <button
              key={tab.key}
              onClick={() => setMode(tab.key)}
              className={`px-3 py-1.5 rounded-md text-[11px] font-semibold transition-all duration-250 ${
                mode === tab.key
                  ? "bg-[rgba(199,91,110,0.15)] text-[#D98090] shadow-[0_0_12px_rgba(199,91,110,0.1)]"
                  : "text-[rgba(245,247,250,0.4)] hover:text-[rgba(245,247,250,0.7)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 px-4 pb-4 relative z-10">
        <motion.div
          className="relative overflow-hidden rounded-xl bg-[#15181D] w-full h-full flex items-center justify-center border border-[rgba(255,255,255,0.05)]"
          style={{ aspectRatio: "16/10" }}
          onHoverStart={() => setHovered(true)}
          onHoverEnd={() => setHovered(false)}
        >
          <motion.div
            className="absolute inset-0 pointer-events-none"
            animate={{
              boxShadow: hovered
                ? "inset 0 0 60px rgba(199,91,110,0.06)"
                : "inset 0 0 60px rgba(199,91,110,0)",
            }}
            transition={{ duration: 0.6 }}
          />

          {mode === "video" ? (
            <motion.div
              animate={{ scale: hovered ? 1.02 : 1 }}
              transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className="w-full h-full"
            >
              <video
                ref={videoRef}
                src={brainMovieUrl}
                autoPlay
                loop
                muted
                playsInline
                className="w-full h-full object-contain"
              />
            </motion.div>
          ) : (
            <motion.img
              src={mode === "mean" ? meanBrainUrl : peakBrainUrl}
              alt={`${mode} brain activation`}
              className="max-h-full max-w-full object-contain"
              animate={{ scale: hovered ? 1.04 : 1 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            />
          )}
        </motion.div>
      </div>
    </div>
  );
}
