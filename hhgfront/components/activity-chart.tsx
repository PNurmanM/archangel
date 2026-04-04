"use client";

import { useState, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { TimePoint } from "@/lib/types";
import { motion } from "framer-motion";

interface ActivityChartProps {
  timeline: TimePoint[];
  systems: string[];
}

/* ── Compact tooltip — only shows selected series ── */
function CustomTooltip({
  active,
  payload,
  label,
  activeSystem,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: number;
  activeSystem: string | null;
}) {
  if (!active || !payload?.length) return null;

  const filtered = payload.filter((entry) => {
    if (activeSystem === null) return entry.name === "Total";
    return entry.name === activeSystem || entry.name === "Total";
  });

  if (!filtered.length) return null;

  return (
    <div className="bg-[#1C2027] border border-[rgba(255,255,255,0.12)] rounded-lg px-3 py-2 shadow-[0_8px_32px_rgba(0,0,0,0.5)]">
      <p className="text-[rgba(245,247,250,0.44)] tabular text-[10px] font-medium mb-1">{label?.toFixed(1)}s</p>
      {filtered.map((entry) => (
        <div key={entry.name} className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: entry.color }} />
            <span className="text-[11px] text-[rgba(245,247,250,0.68)]">{entry.name}</span>
          </div>
          <span className="tabular text-[#F5F7FA] font-semibold text-[11px]">{entry.value.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

export function ActivityChart({ timeline, systems }: ActivityChartProps) {
  const [activeSystem, setActiveSystem] = useState<string | null>(null);

  const data = useMemo(
    () => timeline.map((p) => ({ time: p.time, Total: p.total, ...p.systems })),
    [timeline]
  );

  /* Generate integer-second tick values from the timeline duration */
  const secondTicks = useMemo(() => {
    if (!data.length) return [];
    const maxTime = Math.ceil(data[data.length - 1].time);
    const ticks: number[] = [];
    for (let s = 0; s <= maxTime; s++) ticks.push(s);
    return ticks;
  }, [data]);

  /* Compute summary stats */
  const stats = useMemo(() => {
    if (!data.length) return null;

    const key = activeSystem || "Total";
    const values = data.map((d) => (d as Record<string, number>)[key] ?? 0);
    const peak = Math.max(...values);
    const peakIdx = values.indexOf(peak);
    const peakTime = data[peakIdx]?.time ?? 0;
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);

    const positiveSystems = systems.filter((sys) => {
      const sysValues = data.map((d) => (d as Record<string, number>)[sys] ?? 0);
      const sysAvg = sysValues.reduce((a, b) => a + b, 0) / sysValues.length;
      return sysAvg > 0;
    });

    return { peak, peakTime, avg, min, range: peak - min, activeSystems: positiveSystems.length };
  }, [data, activeSystem, systems]);

  return (
    <div className="card-surface p-5 md:p-6 flex flex-col">
      <div className="flex flex-col gap-4 flex-1">
        <div>
          <span className="section-label">Activity Landscape</span>
          <h3 className="text-base font-semibold text-[#F5F7FA] mt-1">Total Brain Activity</h3>
        </div>

        {/* System toggles — pill chips */}
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setActiveSystem(null)}
            className={`px-3 py-1.5 text-[11px] font-semibold transition-all duration-250 ${
              activeSystem === null ? "pill-chip-active" : "pill-chip"
            }`}
          >
            Total
          </button>
          {systems.map((sys) => (
            <button
              key={sys}
              onClick={() => setActiveSystem(activeSystem === sys ? null : sys)}
              className={`px-3 py-1.5 text-[11px] font-semibold transition-all duration-250 ${
                activeSystem === sys ? "pill-chip-active" : "pill-chip"
              }`}
            >
              {sys}
            </button>
          ))}
        </div>

        {/* Chart */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="h-[260px] md:h-[300px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 14, left: 4, bottom: 20 }}>
              <defs>
                <linearGradient id="grad-total" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#C75B6E" stopOpacity={0.25} />
                  <stop offset="50%" stopColor="#D98090" stopOpacity={0.08} />
                  <stop offset="100%" stopColor="#C75B6E" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="grad-system" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#A04458" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#A04458" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="0" vertical={false} />
              <XAxis
                dataKey="time"
                type="number"
                domain={[0, secondTicks.length > 0 ? secondTicks[secondTicks.length - 1] : "auto"]}
                ticks={secondTicks}
                tick={{ fill: "rgba(245,247,250,0.44)", fontSize: 11, fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${v}s`}
                label={{
                  value: "Time (s)",
                  position: "insideBottom",
                  offset: -12,
                  style: { fill: "rgba(245,247,250,0.36)", fontSize: 10, fontWeight: 600, letterSpacing: "0.05em" },
                }}
              />
              <YAxis
                tick={{ fill: "rgba(245,247,250,0.44)", fontSize: 11, fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
                tickCount={5}
                label={{
                  value: "Activation",
                  angle: -90,
                  position: "insideLeft",
                  offset: 10,
                  style: { fill: "rgba(245,247,250,0.36)", fontSize: 10, fontWeight: 600, letterSpacing: "0.05em" },
                }}
              />
              <Tooltip
                content={<CustomTooltip activeSystem={activeSystem} />}
                cursor={{ stroke: "rgba(199,91,110,0.2)", strokeWidth: 1 }}
              />

              <Area
                type="monotone"
                dataKey="Total"
                stroke="#C75B6E"
                strokeWidth={activeSystem ? 1 : 2.5}
                strokeOpacity={activeSystem ? 0.15 : 1}
                fill="url(#grad-total)"
                fillOpacity={activeSystem ? 0.05 : 1}
                dot={false}
                animationDuration={1000}
              />

              {systems.map((sys) => (
                <Area
                  key={sys}
                  type="monotone"
                  dataKey={sys}
                  stroke="#A04458"
                  strokeWidth={activeSystem === sys ? 2.5 : 1}
                  strokeOpacity={activeSystem === null ? 0 : activeSystem === sys ? 1 : 0.06}
                  fill={activeSystem === sys ? "url(#grad-system)" : "transparent"}
                  fillOpacity={activeSystem === sys ? 1 : 0}
                  dot={false}
                  animationDuration={800}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* ── Stats strip below chart ── */}
        {stats && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.3 }}
            className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1"
          >
            <div className="bg-[#15181D] rounded-xl px-3.5 py-3 border border-[rgba(255,255,255,0.05)]">
              <p className="text-[10px] text-[rgba(245,247,250,0.48)] font-semibold uppercase tracking-wider mb-1">
                Peak
              </p>
              <p className="text-[15px] font-bold tabular text-[#F5F7FA]">
                {stats.peak.toFixed(4)}
              </p>
              <p className="text-[10px] tabular text-[rgba(245,247,250,0.36)] mt-0.5">
                at {stats.peakTime.toFixed(1)}s
              </p>
            </div>
            <div className="bg-[#15181D] rounded-xl px-3.5 py-3 border border-[rgba(255,255,255,0.05)]">
              <p className="text-[10px] text-[rgba(245,247,250,0.48)] font-semibold uppercase tracking-wider mb-1">
                Average
              </p>
              <p className="text-[15px] font-bold tabular text-[#F5F7FA]">
                {stats.avg.toFixed(4)}
              </p>
              <p className="text-[10px] tabular text-[rgba(245,247,250,0.36)] mt-0.5">
                mean activation
              </p>
            </div>
            <div className="bg-[#15181D] rounded-xl px-3.5 py-3 border border-[rgba(255,255,255,0.05)]">
              <p className="text-[10px] text-[rgba(245,247,250,0.48)] font-semibold uppercase tracking-wider mb-1">
                Range
              </p>
              <p className="text-[15px] font-bold tabular text-[#F5F7FA]">
                {stats.range.toFixed(4)}
              </p>
              <p className="text-[10px] tabular text-[rgba(245,247,250,0.36)] mt-0.5">
                max &minus; min
              </p>
            </div>
            <div className="bg-[#15181D] rounded-xl px-3.5 py-3 border border-[rgba(255,255,255,0.05)]">
              <p className="text-[10px] text-[rgba(245,247,250,0.48)] font-semibold uppercase tracking-wider mb-1">
                Active Systems
              </p>
              <p className="text-[15px] font-bold tabular text-[#F5F7FA]">
                {stats.activeSystems}<span className="text-[11px] font-medium text-[rgba(245,247,250,0.36)]"> / {systems.length}</span>
              </p>
              <p className="text-[10px] tabular text-[rgba(245,247,250,0.36)] mt-0.5">
                above baseline
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
