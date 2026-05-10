interface SectionHeaderProps {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export function SectionHeader({ eyebrow, title, description, action }: SectionHeaderProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        gap: 16,
        marginBottom: 24,
      }}
    >
      <div>
        {eyebrow && (
          <p className="text-label" style={{ color: "var(--text-subtle)", marginBottom: 6 }}>
            {eyebrow}
          </p>
        )}
        <h2 className="text-subheading" style={{ margin: 0 }}>
          {title}
        </h2>
        {description && (
          <p
            style={{
              margin: "8px 0 0",
              fontSize: "0.9rem",
              color: "var(--text-muted)",
              fontFamily: "var(--font-thai)",
              lineHeight: 1.6,
            }}
          >
            {description}
          </p>
        )}
      </div>
      {action && <div style={{ flexShrink: 0 }}>{action}</div>}
    </div>
  );
}
