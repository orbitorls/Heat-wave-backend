"use client";

import Link from "next/link";
import { StationSwitcher } from "./StationSwitcher";
import { ThemeToggle } from "./ThemeToggle";

function ShieldIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <path d="M9 2L3 4.5v5c0 3.5 2.5 6 6 7 3.5-1 6-3.5 6-7v-5L9 2Z" fill="white" opacity="0.9"/>
      <path d="M6.5 9l1.8 1.8 3.2-3.2" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" opacity="0.7"/>
    </svg>
  );
}

interface AppShellProps {
  children: React.ReactNode;
  breadcrumb?: React.ReactNode;
}

export function AppShell({ children, breadcrumb }: AppShellProps) {
  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-to-content">
        ข้ามไปเนื้อหาหลัก
      </a>

      <header className="app-topbar">
        <Link href="/" className="app-topbar-logo" aria-label="HeatShield AI — หน้าหลัก">
          <div className="app-topbar-logo-mark" aria-hidden="true">
            <ShieldIcon />
          </div>
          <span className="app-topbar-name">HeatShield</span>
        </Link>

        {breadcrumb && (
          <>
            <div className="app-topbar-sep" aria-hidden="true" />
            <nav aria-label="เส้นทาง">{breadcrumb}</nav>
          </>
        )}

        <div className="app-topbar-spacer" />

        <div className="app-topbar-actions">
          <StationSwitcher />
          <ThemeToggle />
        </div>
      </header>

      <main id="main-content" className="app-content">
        {children}
      </main>

      <footer className="app-footer">
        <p className="app-footer-text">
          HeatShield AI — ข้อมูลจาก TMD · ERA5 · NASA POWER
        </p>
        <p className="app-footer-text" style={{ fontFamily: "var(--font-mono)" }}>
          5 สถานี · Thailand
        </p>
      </footer>
    </div>
  );
}
