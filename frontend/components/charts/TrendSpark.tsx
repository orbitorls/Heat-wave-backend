"use client";

import type { ForecastPoint } from "@/lib/api-types";
import { hiToRiskClass, getTier } from "@/lib/tier";

interface TrendSparkProps {
  points: ForecastPoint[];
  width?: number;
  height?: number;
}

export function TrendSpark({ points, width = 80, height = 28 }: TrendSparkProps) {
  if (!points || points.length < 2) return null;

  const values = points.map((p) => p.heat_index_c);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const xs = points.map((_, i) => (i / (points.length - 1)) * width);
  const ys = values.map((v) => height - ((v - min) / range) * (height - 4) - 2);

  const d = xs
    .map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`)
    .join(" ");

  const lastTier = getTier(hiToRiskClass(values[values.length - 1]));
  const firstTier = getTier(hiToRiskClass(values[0]));
  const trending = values[values.length - 1] > values[0];

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-label={`แนวโน้ม heat index ${trending ? "สูงขึ้น" : "ลดลง"}`}
      style={{ overflow: "visible", flexShrink: 0 }}
    >
      {/* Fill under curve */}
      <path
        d={`${d} L ${xs[xs.length - 1].toFixed(1)} ${height} L 0 ${height} Z`}
        fill={lastTier.color}
        opacity={0.12}
      />
      {/* Line */}
      <path
        d={d}
        fill="none"
        stroke={lastTier.color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Start dot */}
      <circle cx={xs[0]} cy={ys[0]} r={2.5} fill={firstTier.color} />
      {/* End dot */}
      <circle cx={xs[xs.length - 1]} cy={ys[ys.length - 1]} r={3} fill={lastTier.color} />
    </svg>
  );
}
