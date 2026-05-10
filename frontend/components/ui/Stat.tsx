interface StatProps {
  label: string;
  value: string | number;
  unit?: string;
  delta?: string;
  accent?: boolean;
  size?: "sm" | "md" | "lg";
}

const valueSizeMap = {
  sm: "text-data-sm",
  md: "text-data-md",
  lg: "text-data-lg",
} as const;

export function Stat({ label, value, unit, delta, accent, size = "md" }: StatProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span
        className="text-label"
        style={{ color: "var(--text-subtle)", fontSize: "0.72rem" }}
      >
        {label}
      </span>
      <span
        className={valueSizeMap[size]}
        style={{ color: accent ? "var(--accent)" : "var(--text)" }}
      >
        {value}
        {unit && (
          <span
            style={{ fontSize: "0.6em", fontWeight: 500, marginLeft: "0.2em", color: "var(--text-muted)" }}
          >
            {unit}
          </span>
        )}
      </span>
      {delta && (
        <span style={{ fontSize: "0.78rem", color: "var(--text-subtle)", fontFamily: "var(--font-thai)" }}>
          {delta}
        </span>
      )}
    </div>
  );
}
