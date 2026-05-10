interface LowConfidenceBadgeProps {
  reason?: string | null;
}

export function LowConfidenceBadge({ reason }: LowConfidenceBadgeProps) {
  return (
    <span
      title={reason ?? "ข้อมูลล่าสุดอาจไม่เป็นปัจจุบัน"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        padding: "2px 8px",
        borderRadius: "4px",
        fontSize: "0.75rem",
        fontWeight: 500,
        border: "1px solid oklch(0.72 0.15 80)",
        background: "oklch(0.96 0.030 80)",
        color: "oklch(0.38 0.15 80)",
      }}
    >
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden>
        <circle cx="5" cy="5" r="4.5" stroke="currentColor" />
        <line x1="5" y1="3" x2="5" y2="5.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="5" cy="7" r="0.6" fill="currentColor" />
      </svg>
      ความเชื่อมั่นต่ำ
    </span>
  );
}
