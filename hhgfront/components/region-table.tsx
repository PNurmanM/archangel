"use client";

import { BrainRegion } from "@/lib/types";
import { motion } from "framer-motion";

interface RegionListProps {
  regions: BrainRegion[];
}

export function RegionList({ regions }: RegionListProps) {
  const maxScore = Math.max(...regions.map((r) => Math.abs(r.score)));

  return (
    <div className="card-surface p-5">
      <span className="section-label">Top Regions</span>
      <div className="mt-3.5 flex flex-col">
        {regions.map((r, i) => {
          const pct = maxScore > 0 ? (Math.abs(r.score) / maxScore) * 100 : 0;
          return (
            <motion.div
              key={r.rank}
              initial={{ opacity: 0, x: -4 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.12 + i * 0.03, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="flex items-center gap-3 py-2.5 border-b border-[rgba(255,255,255,0.06)] last:border-0 group"
            >
              <span className="text-[11px] tabular text-[rgba(245,247,250,0.30)] w-4 text-right shrink-0 font-semibold group-hover:text-[rgba(245,247,250,0.50)] transition-colors">
                {r.rank}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] text-[rgba(245,247,250,0.85)] truncate leading-snug font-medium group-hover:text-[#F5F7FA] transition-colors">
                  {r.name}
                </p>
                <p className="text-[11px] text-[rgba(245,247,250,0.34)]">{r.label}</p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <div className="w-10 h-[3px] rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{
                      background: "linear-gradient(90deg, #C75B6E, #D98090)",
                      boxShadow: "0 0 6px rgba(199,91,110,0.2)",
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ delay: 0.2 + i * 0.03, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
                <span className="text-[11px] tabular font-semibold text-[rgba(245,247,250,0.60)] w-12 text-right">
                  +{r.score.toFixed(4)}
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
