"use client";

// Custom toast — replaces SweetAlert2, no heavy deps

export type ToastVariant = "success" | "error" | "warning" | "info" | "loading";

const variantConfig: Record<
  ToastVariant,
  { bg: string; border: string; color: string; icon: string }
> = {
  success: {
    bg: "oklch(0.94 0.04 145)",
    border: "oklch(0.55 0.16 145)",
    color: "oklch(0.26 0.14 145)",
    icon: "✓",
  },
  error: {
    bg: "var(--color-risk-critical-bg)",
    border: "var(--color-risk-critical)",
    color: "var(--color-risk-critical-text)",
    icon: "✕",
  },
  warning: {
    bg: "var(--color-risk-high-bg)",
    border: "var(--color-risk-high)",
    color: "var(--color-risk-high-text)",
    icon: "!",
  },
  info: {
    bg: "var(--accent-bg)",
    border: "var(--accent)",
    color: "var(--accent-text)",
    icon: "i",
  },
  loading: {
    bg: "var(--bg-2)",
    border: "var(--border)",
    color: "var(--text-muted)",
    icon: "…",
  },
};

function fire(title: string, variant: ToastVariant = "info") {
  if (typeof window === "undefined") return undefined;
  const cfg = variantConfig[variant];

  // Build DOM nodes without innerHTML to avoid XSS
  const el = document.createElement("div");
  el.setAttribute("role", "alert");
  el.setAttribute("aria-live", "assertive");
  Object.assign(el.style, {
    position: "fixed",
    bottom: "24px",
    right: "24px",
    zIndex: "9999",
    background: cfg.bg,
    border: `1px solid ${cfg.border}`,
    color: cfg.color,
    borderRadius: "var(--radius-lg)",
    padding: "12px 16px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
    boxShadow: "var(--shadow-lg)",
    fontFamily: "var(--font-thai)",
    fontSize: "0.9rem",
    fontWeight: "600",
    maxWidth: "320px",
    cursor: "pointer",
    animation: "fade-up 280ms cubic-bezier(0.16,1,0.3,1) both",
  });

  const iconEl = document.createElement("span");
  Object.assign(iconEl.style, {
    width: "22px",
    height: "22px",
    borderRadius: "50%",
    background: `${cfg.border}33`,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "0.75rem",
    fontWeight: "700",
    flexShrink: "0",
  });
  iconEl.setAttribute("aria-hidden", "true");
  iconEl.textContent = cfg.icon;

  const textEl = document.createElement("span");
  textEl.textContent = title;

  el.appendChild(iconEl);
  el.appendChild(textEl);

  const dismiss = () => {
    el.style.opacity = "0";
    el.style.transition = "opacity 200ms ease";
    setTimeout(() => el.remove(), 200);
  };

  el.addEventListener("click", dismiss);
  document.body.appendChild(el);

  if (variant !== "loading") {
    setTimeout(dismiss, variant === "error" ? 4500 : 3200);
  }

  return { close: dismiss };
}

export const toast = {
  success: (msg: string) => fire(msg, "success"),
  error: (msg: string) => fire(msg, "error"),
  info: (msg: string) => fire(msg, "info"),
  warning: (msg: string) => fire(msg, "warning"),
  loading: (msg = "กำลังดำเนินการ...") => fire(msg, "loading"),
  close: () => {},
};

export function useToast() {
  return toast;
}
