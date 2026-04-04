"use client";

import { motion } from "framer-motion";

interface SummaryPanelProps {
  summary: string;
}

export function SummaryPanel({ summary }: SummaryPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className="card-surface px-6 py-6 md:px-8 md:py-7"
    >
      <div className="flex flex-col md:flex-row gap-5 md:gap-10">
        <div className="shrink-0 md:w-44">
          <span className="section-label">Summary</span>
          <h3 className="text-lg font-semibold text-[#F5F7FA] mt-1.5 leading-tight">
            Brain Response
          </h3>
          <p className="text-[12px] text-[rgba(245,247,250,0.34)] mt-1">
            Interpretation of detected activity patterns
          </p>
        </div>
        <div className="flex-1">
          <p className="text-[14px] leading-[1.85] text-[rgba(245,247,250,0.65)]">
            {summary}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
