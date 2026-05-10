"use client";

import { useMemo } from "react";
import { Group } from "@visx/group";
import { LinePath, Area } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";
import { ParentSize } from "@visx/responsive";
import { curveMonotoneX } from "d3-shape";
import type { ForecastPoint } from "@/lib/api-types";
import { hiToRiskClass, getTier } from "@/lib/tier";

interface ForecastChartProps {
  forecasts: ForecastPoint[];
  lowConfidence?: boolean;
}

function ChartInner({
  forecasts,
  lowConfidence = false,
  width,
  height,
}: ForecastChartProps & { width: number; height: number }) {
  const margin = { top: 16, right: 16, bottom: 36, left: 44 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const points = useMemo(
    () => [...forecasts].sort((a, b) => a.horizon_h - b.horizon_h),
    [forecasts],
  );

  const xScale = useMemo(
    () =>
      scaleLinear({
        domain: [0, Math.max(...points.map((p) => p.horizon_h), 48)],
        range: [0, innerWidth],
      }),
    [points, innerWidth],
  );

  const hiValues = points.map((p) => p.heat_index_c);
  const piLowers = points.map((p) => p.pi_lower ?? p.heat_index_c);
  const piUppers = points.map((p) => p.pi_upper ?? p.heat_index_c);
  const allValues = [...hiValues, ...piLowers, ...piUppers];

  const yScale = useMemo(
    () =>
      scaleLinear({
        domain: [
          Math.min(...allValues) - 1,
          Math.max(...allValues) + 1,
        ],
        range: [innerHeight, 0],
        nice: true,
      }),
    [allValues, innerHeight],
  );

  const primaryTier = getTier(hiToRiskClass(points[0]?.heat_index_c ?? 30));

  if (width < 10 || innerWidth < 10) return null;

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <Group left={margin.left} top={margin.top}>
        {/* PI band */}
        <Area
          data={points}
          x={(p) => xScale(p.horizon_h)}
          y0={(p) => yScale(p.pi_lower ?? p.heat_index_c)}
          y1={(p) => yScale(p.pi_upper ?? p.heat_index_c)}
          curve={curveMonotoneX}
          fill={primaryTier.color}
          fillOpacity={lowConfidence ? 0.08 : 0.14}
          className="animate-fade-in"
        />

        {/* Axis lines */}
        <AxisLeft
          scale={yScale}
          numTicks={4}
          tickFormat={(v) => `${v}°`}
          tickLabelProps={{ fontSize: 11, fill: "var(--text-muted)", textAnchor: "end" }}
          tickLineProps={{ stroke: "var(--border)" }}
          axisLineClassName="stroke-[var(--border)]"
        />
        <AxisBottom
          scale={xScale}
          top={innerHeight}
          numTicks={4}
          tickFormat={(v) => `+${v}h`}
          tickLabelProps={{ fontSize: 11, fill: "var(--text-muted)", textAnchor: "middle" }}
          tickLineProps={{ stroke: "var(--border)" }}
          axisLineClassName="stroke-[var(--border)]"
        />

        {/* Main line */}
        <LinePath
          data={points}
          x={(p) => xScale(p.horizon_h)}
          y={(p) => yScale(p.heat_index_c)}
          curve={curveMonotoneX}
          stroke={primaryTier.color}
          strokeWidth={2.5}
          strokeOpacity={lowConfidence ? 0.5 : 1}
          strokeDasharray={lowConfidence ? "6 3" : undefined}
        />

        {/* Data points */}
        {points.map((p) => {
          const tier = getTier(hiToRiskClass(p.heat_index_c));
          return (
            <circle
              key={p.horizon_h}
              cx={xScale(p.horizon_h)}
              cy={yScale(p.heat_index_c)}
              r={4}
              fill={tier.color}
              stroke="var(--bg)"
              strokeWidth={2}
            />
          );
        })}
      </Group>
    </svg>
  );
}

export function ForecastChart({ forecasts, lowConfidence }: ForecastChartProps) {
  if (!forecasts.length) {
    return (
      <div
        style={{
          height: 180,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text-muted)",
          fontSize: "0.875rem",
        }}
      >
        ไม่มีข้อมูล forecast
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: 200 }}>
      <ParentSize>
        {({ width }) => (
          <ChartInner
            forecasts={forecasts}
            lowConfidence={lowConfidence}
            width={width}
            height={200}
          />
        )}
      </ParentSize>
    </div>
  );
}
