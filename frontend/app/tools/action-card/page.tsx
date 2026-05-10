"use client";

import { useState } from "react";
import { useActionCard } from "@/hooks/useActionCard";
import type { ActionCardRequest, ProfileID, ActivityIntensity, Audience } from "@/lib/api-types";
import { getTier } from "@/lib/tier";
import type { RiskClass } from "@/lib/api-types";
import Link from "next/link";

const PROFILES: { id: ProfileID; label: string }[] = [
  { id: "student_primary", label: "นักเรียนประถมศึกษา" },
  { id: "student_secondary", label: "นักเรียนมัธยมศึกษา" },
  { id: "outdoor_worker", label: "แรงงานกลางแจ้ง" },
  { id: "outdoor_worker_heavy", label: "แรงงานก่อสร้าง/เกษตร" },
  { id: "elderly", label: "ผู้สูงอายุ" },
  { id: "general_adult", label: "ผู้ใหญ่ทั่วไป" },
];

const AUDIENCES: { id: Audience; label: string }[] = [
  { id: "teacher", label: "ครู/ผู้ดูแล" },
  { id: "supervisor", label: "หัวหน้างาน" },
  { id: "parent", label: "ผู้ปกครอง" },
  { id: "worker", label: "ผู้ใช้แรงงาน" },
];

export default function ActionCardPage() {
  const [form, setForm] = useState<ActionCardRequest>({
    temperature_c: 35,
    humidity_rh: 70,
    profile_id: "general_adult",
    activity_intensity: "moderate",
    duration_minutes: 60,
    shade_available: false,
    water_access: true,
    time_of_day_hour: 13,
    acclimatized: false,
    audience: "teacher",
  });

  const { mutate, data, isPending, isError } = useActionCard();

  function update<K extends keyof ActionCardRequest>(k: K, v: ActionCardRequest[K]) {
    setForm((prev) => ({ ...prev, [k]: v }));
  }

  const tier = data ? getTier(data.risk_class as RiskClass) : null;

  return (
    <main id="main-content" style={{ maxWidth: 960, margin: "0 auto", padding: "28px clamp(16px, 4vw, 48px) 80px" }}>
      <Link href="/" className="back-link" style={{ marginBottom: 24 }}>
        ← หน้าหลัก
      </Link>

      <h1 className="text-heading" style={{ color: "var(--text)", marginBottom: 6, fontFamily: "var(--font-thai)", marginTop: 20 }}>
        Action Card
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: 32, fontFamily: "var(--font-thai)", fontSize: "0.94rem" }}>
        สร้างคำแนะนำเฉพาะกลุ่ม — พร้อมพิมพ์แจกจ่าย
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))",
          gap: 28,
          alignItems: "start",
        }}
      >
        {/* ── Form ──────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {/* Audience selector */}
          <div>
            <label className="field-label">กลุ่มผู้รับ</label>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }} className="responsive-grid-2">
              {AUDIENCES.map((a) => (
                <button
                  key={a.id}
                  onClick={() => update("audience", a.id)}
                  className={`toggle-pill${form.audience === a.id ? " selected" : ""}`}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>

          {/* Profile */}
          <div>
            <label className="field-label">กลุ่มผู้ใช้</label>
            <select
              value={form.profile_id}
              onChange={(e) => update("profile_id", e.target.value as ProfileID)}
              className="field-input"
            >
              {PROFILES.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </div>

          {/* Temp + Humidity */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }} className="responsive-grid-2">
            <div>
              <label className="field-label">อุณหภูมิ (°C)</label>
              <input type="number" min={-10} max={60} step={0.5} value={form.temperature_c}
                onChange={(e) => update("temperature_c", Number(e.target.value))}
                className="field-input" />
            </div>
            <div>
              <label className="field-label">ความชื้น (%)</label>
              <input type="number" min={0} max={100} step={1} value={form.humidity_rh}
                onChange={(e) => update("humidity_rh", Number(e.target.value))}
                className="field-input" />
            </div>
          </div>

          {/* Duration slider */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <label className="field-label">ระยะเวลา</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: "0.94rem", color: "var(--accent)" }}>
                {form.duration_minutes} นาที
              </span>
            </div>
            <input type="range" min={0} max={720} step={15} value={form.duration_minutes}
              onChange={(e) => update("duration_minutes", Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--accent)" }} />
          </div>

          {/* Time slider */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <label className="field-label">เวลาของวัน</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: "0.94rem", color: "var(--accent)" }}>
                {String(form.time_of_day_hour).padStart(2, "0")}:00 น.
              </span>
            </div>
            <input type="range" min={0} max={23} step={1} value={form.time_of_day_hour}
              onChange={(e) => update("time_of_day_hour", Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--accent)" }} />
          </div>

          <button
            onClick={() => mutate(form)}
            disabled={isPending}
            className="btn-primary"
          >
            {isPending ? "กำลังสร้าง..." : "สร้าง Action Card"}
          </button>

          {isError && (
            <p style={{ fontSize: "0.882rem", color: "var(--color-risk-critical-text)", fontFamily: "var(--font-thai)" }}>
              ไม่สามารถสร้างได้ — ตรวจสอบค่าที่ป้อน
            </p>
          )}
        </div>

        {/* ── Action Card output ────────────────────────────────── */}
        {!data && !isPending && (
          <div
            style={{
              background: "var(--bg-2)",
              border: "1.5px dashed var(--border)",
              borderRadius: "var(--radius-lg)",
              padding: "56px 24px",
              textAlign: "center",
              color: "var(--text-subtle)",
              fontFamily: "var(--font-thai)",
              fontSize: "0.94rem",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 12,
            }}
          >
            กรอกข้อมูลแล้วกด "สร้าง Action Card"
          </div>
        )}

        {isPending && (
          <div
            style={{
              background: "var(--bg-2)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-lg)",
              padding: 24,
              display: "flex",
              flexDirection: "column",
              gap: 14,
            }}
          >
            <div className="skeleton" style={{ height: 24, width: "60%" }} />
            <div className="skeleton" style={{ height: 16, width: "80%" }} />
            <div className="skeleton" style={{ height: 16, width: "70%" }} />
            <div className="skeleton" style={{ height: 60, marginTop: 8 }} />
            <div className="skeleton" style={{ height: 16, width: "55%" }} />
          </div>
        )}

        {data && tier && (
          <div
            className="action-card-print animate-fade-in"
            style={{
              background: "var(--bg)",
              border: `2px solid ${tier.color}`,
              borderRadius: "var(--radius-lg)",
              overflow: "hidden",
            }}
          >
            {/* Print-only header: logo + station + timestamp */}
            <div
              className="print-only"
              style={{
                display: "none",
                padding: "14px 24px",
                borderBottom: "1px solid var(--border)",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span style={{ fontWeight: 700, fontSize: "0.9rem", fontFamily: "var(--font-thai)" }}>
                HeatShield — ระบบเฝ้าระวังความร้อน
              </span>
              <span style={{ fontSize: "0.75rem", color: "#666", fontFamily: "var(--font-mono)" }}>
                {new Date().toLocaleString("th-TH", { dateStyle: "medium", timeStyle: "short" })}
              </span>
            </div>

            {/* Card header with tier color */}
            <div style={{ background: tier.bg, padding: "20px 24px", borderBottom: `2px solid ${tier.color}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 10 }}>
                <span
                  style={{
                    fontSize: "0.72rem",
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: tier.text,
                    fontFamily: "var(--font-thai)",
                  }}
                >
                  {data.audience}
                </span>
                <span
                  className="tier-chip"
                  style={{ background: tier.color, color: "white", fontSize: "0.78rem" }}
                >
                  {data.risk_class}
                </span>
              </div>
              <h2
                style={{
                  fontSize: "1.1rem",
                  fontWeight: 700,
                  color: tier.text,
                  margin: 0,
                  lineHeight: 1.5,
                  fontFamily: "var(--font-thai)",
                }}
              >
                {data.headline_th}
              </h2>
            </div>

            {/* Card body */}
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 18 }}>
              {/* Immediate actions */}
              <section>
                <h3 className="text-label" style={{ color: "var(--text-muted)", marginBottom: 10 }}>
                  ทำทันที
                </h3>
                <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
                  {data.immediate_actions.map((action, i) => (
                    <li
                      key={i}
                      style={{
                        display: "flex",
                        gap: 10,
                        alignItems: "start",
                        fontSize: "0.94rem",
                        fontFamily: "var(--font-thai)",
                        color: "var(--text)",
                        lineHeight: 1.5,
                      }}
                    >
                      <span
                        style={{
                          width: 22,
                          height: 22,
                          borderRadius: 6,
                          border: `2px solid ${tier.color}`,
                          flexShrink: 0,
                          marginTop: 2,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.65rem",
                          color: tier.color,
                          fontWeight: 700,
                        }}
                      >
                        {i + 1}
                      </span>
                      {action}
                    </li>
                  ))}
                </ul>
              </section>

              {/* Monitoring points */}
              <section>
                <h3 className="text-label" style={{ color: "var(--text-muted)", marginBottom: 10 }}>
                  จุดสังเกต
                </h3>
                <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
                  {data.monitoring_points.map((pt, i) => (
                    <li
                      key={i}
                      style={{
                        fontSize: "0.94rem",
                        fontFamily: "var(--font-thai)",
                        color: "var(--text)",
                        paddingLeft: 18,
                        position: "relative",
                        lineHeight: 1.5,
                      }}
                    >
                      <span style={{ position: "absolute", left: 4, color: tier.color, fontWeight: 700 }}>›</span>
                      {pt}
                    </li>
                  ))}
                </ul>
              </section>

              {/* Escalation box */}
              <div
                style={{
                  padding: "14px 18px",
                  borderRadius: "var(--radius-md)",
                  background: "var(--color-risk-critical-bg)",
                  border: "1.5px solid var(--color-risk-critical)",
                }}
              >
                <p
                  className="text-label"
                  style={{ color: "var(--color-risk-critical-text)", marginBottom: 6 }}
                >
                  เมื่อไรต้องขอความช่วยเหลือ
                </p>
                <p style={{ fontSize: "0.94rem", color: "var(--color-risk-critical-text)", margin: 0, fontFamily: "var(--font-thai)", lineHeight: 1.6 }}>
                  {data.when_to_escalate}
                </p>
              </div>

              {/* Footer meta + actions */}
              <div className="no-print" style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(window.location.href).then(() => {
                      const btn = document.getElementById("copy-link-btn");
                      if (btn) {
                        btn.textContent = "คัดลอกแล้ว ✓";
                        setTimeout(() => { btn.textContent = "คัดลอกลิงก์"; }, 2000);
                      }
                    });
                  }}
                  id="copy-link-btn"
                  className="btn-secondary"
                  style={{ fontSize: "0.855rem" }}
                >
                  คัดลอกลิงก์
                </button>
                <button onClick={() => window.print()} className="btn-secondary" style={{ fontSize: "0.855rem" }}>
                  พิมพ์
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
