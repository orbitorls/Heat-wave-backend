"use client";

import { StationGrid } from "@/components/stations/StationGrid";
import { SectionHeader } from "@/components/ui/SectionHeader";
import Link from "next/link";

const TOOL_CARDS = [
  {
    href: "/tools/risk",
    label: "เช็คความเสี่ยง",
    desc: "ป้อนอุณหภูมิ ความชื้น กลุ่มผู้ใช้ รับคะแนนความเสี่ยงทันที",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M10 3v7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
        <circle cx="10" cy="13.5" r="2.5" fill="currentColor" opacity="0.7"/>
        <path d="M7 3h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
        <path d="M4.5 16.5a7 7 0 1 1 11 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none"/>
      </svg>
    ),
    priority: true,
  },
  {
    href: "/tools/events",
    label: "ตรวจจับ Heatwave",
    desc: "วิเคราะห์คลื่นร้อนจากข้อมูลอุณหภูมิรายวันหลายปี",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M10 2L3 7v7c0 3 3 5 7 6 4-1 7-3 7-6V7L10 2Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" fill="none"/>
        <path d="M7 10l2 2 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
    priority: true,
  },
  {
    href: "/tools/whatif",
    label: "จำลองมาตรการ",
    desc: "เปรียบเทียบผลของร่มเงา น้ำดื่ม หรือการปรับเวลาทำงาน",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M3 10h4l2-6 4 12 2-6h2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
    priority: false,
  },
  {
    href: "/tools/action-card",
    label: "Action Card",
    desc: "สร้างบัตรคำแนะนำสำหรับครู หัวหน้างาน ผู้ปกครอง พร้อมพิมพ์",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <rect x="4" y="3" width="12" height="14" rx="2" stroke="currentColor" strokeWidth="1.6"/>
        <path d="M7 8h6M7 11h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M7 6h2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
    priority: false,
  },
];

export default function HomePage() {
  return (
    <div
      style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: "clamp(32px, 5vw, 56px) clamp(16px, 3vw, 32px) 64px",
      }}
    >
      {/* Hero */}
      <section
        className="animate-fade-up"
        style={{ marginBottom: "clamp(40px, 5vw, 64px)" }}
        aria-labelledby="hero-title"
      >
        <p
          className="text-label"
          style={{ color: "var(--accent)", marginBottom: 12, fontSize: "0.78rem" }}
        >
          Thailand Heat-Health Risk · 5 TMD Stations
        </p>
        <h1
          id="hero-title"
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "clamp(2rem, 5vw, 3rem)",
            fontWeight: 700,
            color: "var(--text)",
            margin: "0 0 16px",
            lineHeight: 1.15,
            letterSpacing: "-0.02em",
            maxWidth: "20ch",
          }}
        >
          ระบบเฝ้าระวัง<wbr />ความร้อน
        </h1>
        <p
          style={{
            fontSize: "clamp(0.95rem, 2vw, 1.05rem)",
            color: "var(--text-muted)",
            maxWidth: "55ch",
            lineHeight: 1.7,
            margin: 0,
            fontFamily: "var(--font-thai)",
          }}
        >
          เช็คความเสี่ยงด้านสุขภาพจากความร้อน คาดการณ์ล่วงหน้า 6–72 ชั่วโมง
          สำหรับโรงเรียนและแรงงานกลางแจ้งทั่วไทย
        </p>
      </section>

      {/* Tool cards — 2+2 grid, priority cards get visual weight */}
      <section
        aria-labelledby="tools-heading"
        style={{ marginBottom: "clamp(48px, 6vw, 72px)" }}
      >
        <SectionHeader eyebrow="เครื่องมือ" title="วิเคราะห์และวางแผน" />

        <div
          className="stagger-children"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: 12,
          }}
        >
          {TOOL_CARDS.map(({ href, label, desc, icon, priority }) => (
            <Link
              key={href}
              href={href}
              style={{ textDecoration: "none", display: "block" }}
            >
              <div
                className="card-hover"
                style={{
                  padding: priority ? "24px" : "20px 22px",
                  borderRadius: "var(--radius-lg)",
                  background: priority ? "var(--accent-bg)" : "var(--bg-2)",
                  border: `1px solid ${priority ? "var(--accent)33" : "var(--border)"}`,
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                  transition: "border-color 150ms, box-shadow 200ms, transform 200ms cubic-bezier(0.16,1,0.3,1)",
                }}
              >
                <span
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: "var(--radius-sm)",
                    background: priority ? "var(--accent)" : "var(--bg-3)",
                    color: priority ? "white" : "var(--text-muted)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                  aria-hidden="true"
                >
                  {icon}
                </span>
                <span
                  style={{
                    fontSize: "1rem",
                    fontWeight: 700,
                    color: "var(--text)",
                    fontFamily: "var(--font-thai)",
                    letterSpacing: "-0.01em",
                    lineHeight: 1.3,
                  }}
                >
                  {label}
                </span>
                <span
                  style={{
                    fontSize: "0.875rem",
                    color: "var(--text-muted)",
                    lineHeight: 1.55,
                    fontFamily: "var(--font-thai)",
                    flex: 1,
                  }}
                >
                  {desc}
                </span>
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 5,
                    fontSize: "0.82rem",
                    fontWeight: 600,
                    color: priority ? "var(--accent)" : "var(--text-subtle)",
                    fontFamily: "var(--font-thai)",
                    marginTop: 4,
                  }}
                >
                  เปิดใช้งาน
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                    <path d="M2 6h8M6.5 2.5 10 6l-3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Station grid */}
      <section aria-labelledby="stations-heading">
        <SectionHeader
          eyebrow="Live Data"
          title="สถานีตรวจวัดอากาศ"
          description="อัพเดททุก 60 วินาที — คลิกสถานีเพื่อดูรายละเอียดและการพยากรณ์"
        />
        <StationGrid />
      </section>
    </div>
  );
}
