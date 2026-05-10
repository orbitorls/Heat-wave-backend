import type { RiskClass } from "@/lib/api-types";
import { TIER_CONFIG } from "@/lib/tier";

interface TierBadgeProps {
  tier: RiskClass;
  size?: "sm" | "md" | "lg";
  showDot?: boolean;
}

const sizeMap = {
  sm: { fontSize: "0.72rem", padding: "3px 9px", minHeight: "24px" },
  md: { fontSize: "0.82rem", padding: "4px 12px", minHeight: "28px" },
  lg: { fontSize: "0.95rem", padding: "6px 16px", minHeight: "34px" },
} as const;

export function TierBadge({ tier, size = "md", showDot = false }: TierBadgeProps) {
  const cfg = TIER_CONFIG[tier];
  const sz = sizeMap[size];

  return (
    <span
      className="tier-chip"
      style={{
        background: cfg.bg,
        color: cfg.text,
        border: `1.5px solid ${cfg.color}`,
        fontSize: sz.fontSize,
        padding: sz.padding,
        minHeight: sz.minHeight,
        gap: showDot ? 6 : undefined,
      }}
      aria-label={`ระดับความเสี่ยง: ${cfg.label}`}
    >
      {showDot && (
        <span
          style={{ width: 7, height: 7, borderRadius: "50%", background: cfg.color, flexShrink: 0 }}
          aria-hidden="true"
        />
      )}
      {cfg.label}
    </span>
  );
}
