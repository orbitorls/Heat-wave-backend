"use client";

import type { RiskClass } from "@/lib/api-types";
import { getTier, TIER_CONFIG } from "@/lib/tier";
import { formatScore } from "@/lib/format";

interface RiskMeterProps {
  score: number;           // 0–100
  riskClass: RiskClass;
  showLabel?: boolean;
}

export function RiskMeter({ score, riskClass, showLabel = true }: RiskMeterProps) {
  const tier = getTier(riskClass);
  const pct = Math.min(100, Math.max(0, score));

  return (
    <div style={{ width: "100%" }}>
      {showLabel && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 6,
            alignItems: "baseline",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "1.5rem",
              fontWeight: 700,
              color: tier.text,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {formatScore(score)}
          </span>
          <span
            style={{
              fontSize: "0.875rem",
              fontWeight: 600,
              color: tier.text,
              padding: "2px 8px",
              borderRadius: 4,
              background: tier.bg,
              border: `1px solid ${tier.color}`,
            }}
          >
            {tier.label}
          </span>
        </div>
      )}
      {/* Track */}
      <div
        style={{
          position: "relative",
          height: 8,
          borderRadius: 4,
          background: "var(--bg-3)",
          overflow: "hidden",
        }}
        role="meter"
        aria-valuenow={score}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`ความเสี่ยง: ${score}/100 (${tier.label})`}
      >
        {/* Tier zones */}
        {Object.entries(TIER_CONFIG).map(([cls, cfg], i) => {
          const zones = [
            [0, 25],
            [25, 50],
            [50, 75],
            [75, 100],
          ] as [number, number][];
          const [from, to] = zones[i];
          return (
            <div
              key={cls}
              style={{
                position: "absolute",
                top: 0,
                left: `${from}%`,
                width: `${to - from}%`,
                height: "100%",
                background: cfg.color,
                opacity: 0.18,
              }}
            />
          );
        })}
        {/* Fill bar */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            height: "100%",
            width: `${pct}%`,
            background: tier.barColor,
            borderRadius: 4,
            transition: "width 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 4,
          fontSize: "0.7rem",
          color: "var(--text-subtle)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span>0</span>
        <span>100</span>
      </div>
    </div>
  );
}
