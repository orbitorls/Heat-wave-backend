import type { HTMLAttributes } from "react";

interface CardProps extends HTMLAttributes<HTMLElement> {
  as?: "div" | "article" | "section";
  tone?: "default" | "data" | "danger" | "tier";
  density?: "compact" | "normal" | "loose";
  toneColor?: string;
}

const paddingMap = { compact: "16px 20px", normal: "20px 24px", loose: "28px 32px" } as const;

export function Card({
  as: Tag = "div",
  tone = "default",
  density = "normal",
  toneColor,
  style,
  className = "",
  children,
  ...rest
}: CardProps) {
  const base: React.CSSProperties = {
    background: tone === "data" ? "var(--bg-2)" : tone === "danger" ? toneColor ?? "var(--bg-2)" : "var(--bg-2)",
    border: `1px solid ${tone === "tier" && toneColor ? `${toneColor}66` : "var(--border)"}`,
    borderRadius: "var(--radius-lg)",
    padding: paddingMap[density],
    ...(tone === "danger" && toneColor ? { borderColor: `${toneColor}88` } : {}),
    ...style,
  };

  return (
    <Tag className={`ui-card ${className}`} style={base} {...rest}>
      {children}
    </Tag>
  );
}
