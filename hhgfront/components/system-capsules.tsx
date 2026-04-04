"use client";

import { SystemScore } from "@/lib/types";
import { motion } from "framer-motion";

interface SystemCapsulesProps {
  scores: SystemScore[];
}

export function SystemCapsules({ scores }: SystemCapsulesProps) {
  const sorted = [...scores].sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  const maxAbs = Math.max(...sorted.map((s) => Math.abs(s.score)));

  return (
    <div className="card-surface p-5">
      <span className="section-label">System Signals</span>
      <div className="mt-3.5 flex flex-col gap-2.5">
        {sorted.map((s, i) => {
          const pct = maxAbs > 0 ? (Math.abs(s.score) / maxAbs) * 100 : 0;
          const isNeg = s.score < 0;
          return (
            <motion.div
              key={s.system}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 + i * 0.035, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="flex items-center gap-3 group"
            >
              <span className="text-[12px] text-[rgba(245,247,250,0.60)] w-[140px] shrink-0 truncate group-hover:text-[rgba(245,247,250,0.85)] transition-colors">
                {s.system}
              </span>
              <div className="flex-1 h-[5px] rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{
                    background: isNeg
                      ? "linear-gradient(90deg, #617086, #8792A2)"
                      : "linear-gradient(90deg, #C75B6E, #D98090)",
                    boxShadow: isNeg
                      ? "none"
                      : `0 0 8px rgba(199,91,110,${pct > 50 ? 0.3 : 0.15})`,
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ delay: 0.2 + i * 0.04, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
              <span className={`text-[11px] tabular font-semibold w-14 text-right ${
                isNeg ? "text-[#8792A2]" : "text-[rgba(245,247,250,0.72)]"
              }`}>
                {s.score >= 0 ? "+" : ""}{s.score.toFixed(4)}
              </span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
