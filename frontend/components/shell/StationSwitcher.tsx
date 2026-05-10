"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { STATIONS_FALLBACK } from "@/lib/stations";

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      aria-hidden="true"
      style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 150ms ease" }}
    >
      <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function MapPinIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <circle cx="6" cy="5" r="2" fill="currentColor" opacity="0.7"/>
      <path d="M6 1a4 4 0 0 1 4 4c0 3-4 6.5-4 6.5S2 8 2 5a4 4 0 0 1 4-4Z" stroke="currentColor" strokeWidth="1.2" fill="none"/>
    </svg>
  );
}

export function StationSwitcher() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const activeId = pathname.startsWith("/station/")
    ? pathname.split("/station/")[1]?.split("/")[0]
    : null;
  const activeStation = STATIONS_FALLBACK.find((s) => s.station_id === activeId);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div className="station-switcher" ref={ref}>
      <button
        className="station-switcher-btn"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label="เลือกสถานี"
      >
        <MapPinIcon />
        <span className="station-switcher-label">
          {activeStation ? activeStation.name_th.split(" ")[0] : "สถานี"}
        </span>
        <ChevronIcon open={open} />
      </button>

      {open && (
        <div
          className="station-switcher-menu"
          role="listbox"
          aria-label="รายการสถานี"
        >
          {STATIONS_FALLBACK.map((s) => (
            <Link
              key={s.station_id}
              href={`/station/${s.station_id}`}
              className={`station-switcher-item ${s.station_id === activeId ? "active" : ""}`}
              role="option"
              aria-selected={s.station_id === activeId}
              onClick={() => setOpen(false)}
            >
              <span
                className="station-switcher-dot"
                style={{ background: s.station_id === activeId ? "var(--accent)" : "var(--border-strong)" }}
              />
              <span style={{ flex: 1 }}>{s.name_th}</span>
              <span style={{ fontSize: "0.75rem", color: "var(--text-subtle)", fontFamily: "var(--font-mono)" }}>
                {s.station_id}
              </span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
