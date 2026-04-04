"use client";

import { motion } from "framer-motion";
import { AlertLevel } from "@/lib/types";

interface SnapshotRailProps {
  totalActivity: number;
  alertLevel: AlertLevel;
  alertLabel: string;
  dominantSystem: string;
  engagement: string;
}

const STATUS_COLORS: Record<AlertLevel, string> = {
  NORMAL: "#6FD0A3",
  WATCH: "#F4B267",
  ALERT: "#FF8C7A",
};

const STATUS_CLASS: Record<AlertLevel, string> = {
  NORMAL: "status-normal",
  WATCH: "status-watch",
  ALERT: "status-alert",
};

export function SnapshotRail({
  totalActivity,
  alertLevel,
  alertLabel,
  dominantSystem,
  engagement,
}: SnapshotRailProps) {
  const ease = [0.16, 1, 0.3, 1] as const;

  return (
    <div className="card-surface flex flex-col h-full px-5 py-4">
      <span className="section-label mb-5">Snapshot</span>

      <div className="flex flex-col gap-5 flex-1">
        {/* Total Activity */}
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.5, ease }}
        >
          <p className="text-[11px] text-[rgba(245,247,250,0.56)] mb-1 font-medium">Total Activity</p>
          <p className="text-3xl font-bold tabular text-[#F5F7FA] tracking-tight">
            {totalActivity.toFixed(4)}
          </p>
        </motion.div>

        {/* Engagement */}
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25, duration: 0.5, ease }}
        >
          <p className="text-[11px] text-[rgba(245,247,250,0.56)] mb-1 font-medium">Engagement</p>
          <p className="text-sm font-semibold text-[rgba(245,247,250,0.85)]">{engagement}</p>
        </motion.div>

        {/* State */}
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.5, ease }}
        >
          <p className="text-[11px] text-[rgba(245,247,250,0.56)] mb-1.5 font-medium">State</p>
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[12px] font-semibold ${STATUS_CLASS[alertLevel]}`}>
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: STATUS_COLORS[alertLevel] }}
            />
            {alertLabel}
          </span>
        </motion.div>

        {/* Dominant system */}
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45, duration: 0.5, ease }}
          className="mt-auto"
        >
          <p className="text-[11px] text-[rgba(245,247,250,0.56)] mb-1 font-medium">Dominant System</p>
          <p className="text-[13px] text-[rgba(245,247,250,0.65)] font-medium">{dominantSystem}</p>
        </motion.div>
      </div>
    </div>
  );
}
