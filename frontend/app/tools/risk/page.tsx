"use client";

import { useState, useDeferredValue, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { RiskMeter } from "@/components/charts/RiskMeter";
import { TierBadge } from "@/components/ui/TierBadge";
import { Stat } from "@/components/ui/Stat";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { getTier } from "@/lib/tier";
import type { RiskScoreRequest, ProfileID, ActivityIntensity, RiskClass } from "@/lib/api-types";
import Link from "next/link";
import { toast } from "@/components/ui/Toast";

const PROFILES: { id: ProfileID; label: string }[] = [
  { id: "student_primary", label: "นักเรียนประถมศึกษา" },
  { id: "student_secondary", label: "นักเรียนมัธยมศึกษา" },
  { id: "student_university", label: "นิสิต/นักศึกษา" },
  { id: "outdoor_worker", label: "แรงงานกลางแจ้งทั่วไป" },
  { id: "outdoor_worker_heavy", label: "แรงงานก่อสร้าง/เกษตร" },
  { id: "athlete", label: "นักกีฬา/ผู้เล่นกีฬา" },
  { id: "elderly", label: "ผู้สูงอายุ" },
  { id: "general_adult", label: "ผู้ใหญ่ทั่วไป" },
];

const ACTIVITIES: { id: ActivityIntensity; label: string }[] = [
  { id: "rest", label: "พัก/นั่ง" },
  { id: "low", label: "เบา (เดิน)" },
  { id: "moderate", label: "ปานกลาง (พลศึกษา)" },
  { id: "high", label: "หนัก (วิ่ง)" },
  { id: "very_high", label: "หนักมาก (แข่งขัน)" },
];

export default function RiskPage() {
  const [form, setForm] = useState<RiskScoreRequest>({
    temperature_c: 35,
    humidity_rh: 70,
    profile_id: "general_adult",
    activity_intensity: "moderate",
    duration_minutes: 60,
    shade_available: false,
    water_access: true,
    time_of_day_hour: 13,
    acclimatized: false,
  });

  const deferredForm = useDeferredValue(form);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["risk", deferredForm],
    queryFn: () => api.riskScore(deferredForm),
    staleTime: 10_000,
    retry: 1,
  });

  const tier = data ? getTier(data.risk_class as RiskClass) : null;

  useEffect(() => {
    if (isError) {
      toast.error("ไม่สามารถเช็คความเสี่ยงได้ — ลองปรับค่าใหม่อีกครั้ง");
    }
  }, [isError]);

  function update<K extends keyof RiskScoreRequest>(k: K, v: RiskScoreRequest[K]) {
    setForm((prev) => ({ ...prev, [k]: v }));
  }

  return (
    <main
      id="main-content"
      style={{
        maxWidth: 860,
        margin: "0 auto",
        padding: "28px clamp(16px, 4vw, 48px) 80px",
      }}
    >
      <Link href="/" className="back-link" style={{ marginBottom: 24 }}>
        ← หน้าหลัก
      </Link>

      <div style={{ marginTop: 20, marginBottom: 32 }}>
        <SectionHeader
          eyebrow="เครื่องมือ"
          title="เช็คความเสี่ยง"
          description="ปรับตัวแปรด้านล่างแล้วดูคะแนนความเสี่ยงทันที"
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
          gap: 28,
          alignItems: "start",
        }}
      >
        {/* Form */}
        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          {/* Temperature + Humidity */}
          <div
            className="responsive-grid-2"
            style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}
          >
            <div>
              <label
                className="text-label"
                style={{ display: "block", marginBottom: 6, color: "var(--text-muted)" }}
              >
                อุณหภูมิ (°C)
              </label>
              <input
                type="number"
                min={-10}
                max={60}
                step={0.5}
                value={form.temperature_c}
                onChange={(e) => update("temperature_c", Number(e.target.value))}
                className="field-input"
              />
            </div>
            <div>
              <label
                className="text-label"
                style={{ display: "block", marginBottom: 6, color: "var(--text-muted)" }}
              >
                ความชื้นสัมพัทธ์ (%)
              </label>
              <input
                type="number"
                min={0}
                max={100}
                step={1}
                value={form.humidity_rh}
                onChange={(e) => update("humidity_rh", Number(e.target.value))}
                className="field-input"
              />
            </div>
          </div>

          {/* Profile */}
          <div>
            <label
              className="text-label"
              style={{ display: "block", marginBottom: 6, color: "var(--text-muted)" }}
            >
              กลุ่มผู้ใช้
            </label>
            <select
              value={form.profile_id}
              onChange={(e) => update("profile_id", e.target.value as ProfileID)}
              className="field-input"
            >
              {PROFILES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          {/* Activity */}
          <div>
            <label
              className="text-label"
              style={{ display: "block", marginBottom: 6, color: "var(--text-muted)" }}
            >
              ระดับกิจกรรม
            </label>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {ACTIVITIES.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  onClick={() => update("activity_intensity", a.id)}
                  className={`toggle-pill${form.activity_intensity === a.id ? " selected" : ""}`}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>

          {/* Duration slider */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <label
                className="text-label"
                style={{ color: "var(--text-muted)" }}
              >
                ระยะเวลา
              </label>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontWeight: 700,
                  fontSize: "1rem",
                  color: "var(--accent)",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {form.duration_minutes} นาที
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={720}
              step={15}
              value={form.duration_minutes}
              onChange={(e) => update("duration_minutes", Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--accent)", height: 6 }}
              aria-label="ระยะเวลา"
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.72rem",
                color: "var(--text-subtle)",
                marginTop: 2,
              }}
            >
              <span>0 นาที</span>
              <span>720 นาที</span>
            </div>
          </div>

          {/* Time of day slider */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <label
                className="text-label"
                style={{ color: "var(--text-muted)" }}
              >
                เวลาของวัน
              </label>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontWeight: 700,
                  fontSize: "1rem",
                  color: "var(--accent)",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {String(form.time_of_day_hour).padStart(2, "0")}:00 น.
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={23}
              step={1}
              value={form.time_of_day_hour}
              onChange={(e) => update("time_of_day_hour", Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--accent)", height: 6 }}
              aria-label="เวลาของวัน"
            />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.72rem",
                color: "var(--text-subtle)",
                marginTop: 2,
              }}
            >
              <span>00:00</span>
              <span>23:00</span>
            </div>
          </div>

          {/* Environment toggles */}
          <div>
            <span
              className="text-label"
              style={{ display: "block", marginBottom: 8, color: "var(--text-muted)" }}
            >
              สภาพแวดล้อม
            </span>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              {(
                [
                  { key: "shade_available" as const, label: "มีร่มเงา" },
                  { key: "water_access" as const, label: "มีน้ำดื่ม" },
                  { key: "acclimatized" as const, label: "ชินร้อน" },
                ] as const
              ).map(({ key, label }) => (
                <label
                  key={key}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    cursor: "pointer",
                    padding: "10px 14px",
                    minHeight: 44,
                    borderRadius: "var(--radius-md)",
                    border: `1.5px solid ${form[key] ? "var(--accent)" : "var(--border)"}`,
                    background: form[key] ? "var(--accent-bg)" : "var(--bg-2)",
                    fontSize: "0.94rem",
                    color: form[key] ? "var(--accent-text)" : "var(--text)",
                    fontFamily: "var(--font-thai)",
                    fontWeight: form[key] ? 700 : 500,
                    transition: "border-color 150ms, background 150ms",
                    userSelect: "none",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={form[key] as boolean}
                    onChange={(e) => update(key, e.target.checked)}
                    style={{ accentColor: "var(--accent)", width: 18, height: 18 }}
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Result panel — sticky, tone follows tier */}
        <div
          style={{
            background: "var(--bg-2)",
            border: `1.5px solid ${tier ? tier.color + "55" : "var(--border)"}`,
            borderRadius: "var(--radius-lg)",
            overflow: "hidden",
            position: "sticky",
            top: "calc(var(--shell-top-h, 56px) + 16px)",
          }}
        >
          {/* Header band — color shifts with tier */}
          <div
            style={{
              padding: "14px 22px",
              background: tier ? tier.bg : "var(--bg-3)",
              borderBottom: `1.5px solid ${tier ? tier.color + "55" : "var(--border)"}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <h2
              style={{
                fontSize: "0.94rem",
                fontWeight: 700,
                margin: 0,
                color: tier ? tier.text : "var(--text-muted)",
                fontFamily: "var(--font-thai)",
              }}
            >
              ผลลัพธ์
            </h2>
            {data && tier && (
              <TierBadge tier={data.risk_class as RiskClass} size="sm" />
            )}
          </div>

          <div style={{ padding: "22px" }}>
            {isLoading && (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div className="skeleton" style={{ height: 20, width: "65%" }} />
                <div className="skeleton" style={{ height: 20, width: "45%" }} />
                <div className="skeleton" style={{ height: 60, marginTop: 8 }} />
              </div>
            )}

            {isError && !isLoading && (
              <div
                className="animate-shake"
                style={{
                  padding: "14px 18px",
                  borderRadius: "var(--radius-md)",
                  background: "var(--color-risk-critical-bg)",
                  color: "var(--color-risk-critical-text)",
                  fontFamily: "var(--font-thai)",
                  fontSize: "0.94rem",
                  display: "flex",
                  gap: 10,
                  alignItems: "center",
                  border: "1px solid var(--color-risk-critical)",
                }}
              >
                <span aria-hidden="true" style={{ fontWeight: 700, flexShrink: 0 }}>✕</span>
                ไม่สามารถคำนวณได้ — ตรวจสอบค่าที่ป้อน
              </div>
            )}

            {data && tier && (
              <div
                className="animate-fade-in"
                style={{ display: "flex", flexDirection: "column", gap: 20 }}
              >
                <RiskMeter score={data.score} riskClass={data.risk_class as RiskClass} />

                {/* HI stat tile */}
                <div
                  style={{
                    padding: "14px 16px",
                    borderRadius: "var(--radius-md)",
                    background: tier.bg,
                    border: `1px solid ${tier.color}44`,
                  }}
                >
                  <Stat
                    label="Heat Index"
                    value={data.heat_index_c.toFixed(1)}
                    unit="°C"
                    accent
                    size="lg"
                  />
                </div>

                {/* Dominant factors */}
                {data.dominant_factors.length > 0 && (
                  <div>
                    <p
                      className="text-label"
                      style={{ color: "var(--text-muted)", marginBottom: 8 }}
                    >
                      ปัจจัยเสี่ยงหลัก
                    </p>
                    <ul
                      style={{
                        listStyle: "none",
                        padding: 0,
                        margin: 0,
                        display: "flex",
                        flexDirection: "column",
                        gap: 6,
                      }}
                    >
                      {data.dominant_factors.map((f) => (
                        <li
                          key={f}
                          style={{
                            padding: "8px 14px",
                            borderRadius: "var(--radius-sm)",
                            background: tier.bg,
                            fontSize: "0.882rem",
                            color: tier.text,
                            fontFamily: "var(--font-thai)",
                            fontWeight: 500,
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                          }}
                        >
                          <span aria-hidden="true">▸</span> {f}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {data.conservative_applied && (
                  <p
                    style={{
                      fontSize: "0.765rem",
                      color: "var(--text-subtle)",
                      fontFamily: "var(--font-thai)",
                    }}
                  >
                    * ใช้การคำนวณแบบปกติ
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
