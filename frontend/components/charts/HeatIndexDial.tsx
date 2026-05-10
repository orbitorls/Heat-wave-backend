"use client";

import { getTier, hiToRiskClass } from "@/lib/tier";
import { formatHI } from "@/lib/format";

interface HeatIndexDialProps {
  hiC: number;
  size?: number;
}

export function HeatIndexDial({ hiC, size = 120 }: HeatIndexDialProps) {
  const riskClass = hiToRiskClass(hiC);
  const tier = getTier(riskClass);

  // Gauge arc: 0°C=0%, 50°C=100%
  const pct = Math.min(1, Math.max(0, (hiC - 20) / 35));
  const r = (size / 2) * 0.72;
  const cx = size / 2;
  const cy = size / 2 + size * 0.08;
  const startAngle = Math.PI * 0.8;
  const endAngle = Math.PI * 2.2;
  const range = endAngle - startAngle;

  function arc(from: number, to: number): string {
    const x1 = cx + r * Math.cos(from);
    const y1 = cy + r * Math.sin(from);
    const x2 = cx + r * Math.cos(to);
    const y2 = cy + r * Math.sin(to);
    const large = to - from > Math.PI ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  }

  const fillAngle = startAngle + range * pct;

  return (
    <div style={{ display: "inline-block", position: "relative", width: size, height: size * 0.85 }}>
      <svg width={size} height={size * 0.85} viewBox={`0 0 ${size} ${size * 0.85}`} aria-hidden>
        {/* Track */}
        <path
          d={arc(startAngle, endAngle)}
          fill="none"
          stroke={`oklch(0.88 0.008 55)`}
          strokeWidth={size * 0.08}
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d={arc(startAngle, fillAngle)}
          fill="none"
          stroke={tier.color}
          strokeWidth={size * 0.08}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 0.4s ease" }}
        />
      </svg>
      {/* Center label */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          paddingTop: size * 0.08,
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: size * 0.22,
            fontWeight: 700,
            color: tier.text,
            lineHeight: 1,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {formatHI(hiC)}
        </span>
        <span
          style={{
            fontSize: size * 0.11,
            color: "var(--text-muted)",
            marginTop: 2,
          }}
        >
          ค่า
        </span>
      </div>
    </div>
  );
}
