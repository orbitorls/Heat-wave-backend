"use client";

import { useState, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { WhatIfRequest, ProfileID, ActivityIntensity, InterventionIn } from "@/lib/api-types";
import { getTier } from "@/lib/tier";
import type { RiskClass } from "@/lib/api-types";
import Link from "next/link";
import { toast } from "@/components/ui/Toast";

const INTERVENTIONS = [
  { type: "shift_time", label: "เลื่อนเวลา (+2 ชม.)", config: { shift_hours: 2 } },
  { type: "add_shade", label: "เพิ่มร่มเงา", config: { add_shade: true } },
  { type: "add_water", label: "เพิ่มน้ำดื่ม", config: { add_water: true } },
  { type: "reduce_duration", label: "ลดเวลา 30%", config: { reduce_duration_by: 30 } },
  { type: "add_break", label: "พัก 20 นาที", config: { break_every_minutes: 20 } },
  { type: "reduce_intensity", label: "ลดความหนักเป็นปานกลาง", config: { new_intensity: "moderate" } },
] as const;

export default function WhatIfPage() {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [selectedInterventions, setSelectedInterventions] = useState<Set<string>>(new Set());

  const [baseForm, setBaseForm] = useState({
    temperature_c: 36,
    humidity_rh: 75,
    profile_id: "general_adult" as ProfileID,
    activity_intensity: "high" as ActivityIntensity,
    duration_minutes: 120,
    shade_available: false,
    water_access: true,
    time_of_day_hour: 13,
    acclimatized: false,
  });

  const { mutate, data, isPending, isError } = useMutation({
    mutationFn: (req: WhatIfRequest) => api.whatIf(req),
    onSuccess: (result) => {
      setStep(3);
      if (result.best_intervention) {
        toast.success(`มาตรการ "${result.best_intervention}" ลดความเสี่ยงได้มากที่สุด`);
      } else {
        toast.success("จำลองเสร็จแล้ว");
      }
    },
    onError: () => {
      toast.error("จำลองไม่สำเร็จ — ลองอีกครั้ง");
    },
  });

  function toggleIntervention(type: string) {
    setSelectedInterventions((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }

  function buildInterventions(): InterventionIn[] {
    return INTERVENTIONS.filter((iv) => selectedInterventions.has(iv.type)).map((iv) => ({
      intervention_type: iv.type,
      ...iv.config,
    }));
  }

  function handleRun() {
    const interventions = buildInterventions();
    if (!interventions.length) return;
    mutate({ ...baseForm, interventions });
  }

  const STEPS = [
    { n: 1, label: "สภาพแวดล้อม" },
    { n: 2, label: "มาตรการ" },
    { n: 3, label: "ผลเปรียบเทียบ" },
  ] as const;

  return (
    <main id="main-content" style={{ maxWidth: 860, margin: "0 auto", padding: "28px clamp(16px, 4vw, 48px) 80px" }}>
      <Link href="/" className="back-link" style={{ marginBottom: 24 }}>
        ← หน้าหลัก
      </Link>

      <h1 className="text-heading" style={{ color: "var(--text)", marginBottom: 6, fontFamily: "var(--font-thai)", marginTop: 20 }}>
        จำลองมาตรการลดความเสี่ยง
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: 32, fontFamily: "var(--font-thai)", fontSize: "0.94rem" }}>
        เปรียบเทียบมาตรการลดความเสี่ยง
      </p>

      {/* ── Step pill indicator ──────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 32,
          padding: "14px 18px",
          background: "var(--bg-2)",
          borderRadius: "var(--radius-lg)",
          border: "1px solid var(--border)",
          flexWrap: "wrap",
        }}
      >
        {STEPS.map(({ n, label }) => {
          const done = step > n;
          const active = step === n;
          return (
            <button
              key={n}
              onClick={() => done && setStep(n as 1 | 2 | 3)}
              disabled={!done}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 16px",
                minHeight: 40,
                borderRadius: "var(--radius-md)",
                border: `1.5px solid ${active ? "var(--accent)" : done ? "var(--border)" : "transparent"}`,
                background: active ? "var(--accent)" : done ? "var(--bg-3)" : "transparent",
                color: active ? "white" : done ? "var(--text-muted)" : "var(--text-subtle)",
                fontSize: "0.94rem",
                fontWeight: active ? 700 : 500,
                fontFamily: "var(--font-thai)",
                cursor: done ? "pointer" : "default",
                transition: "all 150ms",
              }}
            >
              <span
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: "50%",
                  background: active ? "white" : done ? "var(--accent)" : "var(--border)",
                  color: active ? "var(--accent)" : done ? "white" : "var(--text-subtle)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  flexShrink: 0,
                }}
              >
                {done ? "✓" : n}
              </span>
              {label}
            </button>
          );
        })}
      </div>

      {/* ── Step 1: Base conditions ──────────────────────────────── */}
      {step === 1 && (
        <div className="animate-fade-in" style={{ maxWidth: 540 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }} className="responsive-grid-2">
            <div>
              <label className="field-label">อุณหภูมิ (°C)</label>
              <input
                type="number" min={-10} max={60} step={0.5}
                value={baseForm.temperature_c}
                onChange={(e) => setBaseForm((p) => ({ ...p, temperature_c: Number(e.target.value) }))}
                className="field-input"
              />
            </div>
            <div>
              <label className="field-label">ความชื้นสัมพัทธ์ (%)</label>
              <input
                type="number" min={0} max={100} step={1}
                value={baseForm.humidity_rh}
                onChange={(e) => setBaseForm((p) => ({ ...p, humidity_rh: Number(e.target.value) }))}
                className="field-input"
              />
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <label className="field-label">ระยะเวลา</label>
                <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: "0.94rem", color: "var(--accent)" }}>
                  {baseForm.duration_minutes} นาที
                </span>
              </div>
              <input
                type="range" min={1} max={720} step={15}
                value={baseForm.duration_minutes}
                onChange={(e) => setBaseForm((p) => ({ ...p, duration_minutes: Number(e.target.value) }))}
                style={{ width: "100%", accentColor: "var(--accent)" }}
              />
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <label className="field-label">เวลาของวัน</label>
                <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: "0.94rem", color: "var(--accent)" }}>
                  {String(baseForm.time_of_day_hour).padStart(2, "0")}:00 น.
                </span>
              </div>
              <input
                type="range" min={0} max={23} step={1}
                value={baseForm.time_of_day_hour}
                onChange={(e) => setBaseForm((p) => ({ ...p, time_of_day_hour: Number(e.target.value) }))}
                style={{ width: "100%", accentColor: "var(--accent)" }}
              />
            </div>
          </div>
          <button onClick={() => setStep(2)} className="btn-primary">
            ถัดไป: เลือกมาตรการ →
          </button>
        </div>
      )}

      {/* ── Step 2: Interventions ────────────────────────────────── */}
      {step === 2 && (
        <div className="animate-fade-in" style={{ maxWidth: 640 }}>
          <p style={{ color: "var(--text-muted)", marginBottom: 18, fontFamily: "var(--font-thai)", fontSize: "0.94rem" }}>
            เลือกมาตรการที่ต้องการเปรียบเทียบ (เลือกได้หลายอย่าง)
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 24 }}>
            {INTERVENTIONS.map((iv) => {
              const sel = selectedInterventions.has(iv.type);
              return (
                <button
                  key={iv.type}
                  onClick={() => toggleIntervention(iv.type)}
                  style={{
                    padding: "14px 16px",
                    minHeight: 54,
                    borderRadius: "var(--radius-md)",
                    border: `1.5px solid ${sel ? "var(--accent)" : "var(--border)"}`,
                    background: sel ? "var(--accent-bg)" : "var(--bg-2)",
                    color: sel ? "var(--accent-text)" : "var(--text)",
                    fontSize: "0.94rem",
                    fontWeight: sel ? 700 : 500,
                    cursor: "pointer",
                    textAlign: "left",
                    fontFamily: "var(--font-thai)",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    transition: "all 150ms",
                  }}
                >
                  <span
                    style={{
                      width: 22,
                      height: 22,
                      borderRadius: 6,
                      border: `2px solid ${sel ? "var(--accent)" : "var(--border)"}`,
                      background: sel ? "var(--accent)" : "transparent",
                      color: "white",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "0.75rem",
                      flexShrink: 0,
                      transition: "all 150ms",
                    }}
                  >
                    {sel ? "✓" : ""}
                  </span>
                  {iv.label}
                </button>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button onClick={() => setStep(1)} className="btn-secondary">
              ← ย้อนกลับ
            </button>
            <button
              onClick={handleRun}
              disabled={isPending || selectedInterventions.size === 0}
              className="btn-primary"
            >
              {isPending ? "กำลังจำลอง..." : "เรียกใช้การจำลอง →"}
            </button>
          </div>
          {isError && (
            <p style={{ color: "var(--color-risk-critical-text)", fontSize: "0.882rem", marginTop: 14, fontFamily: "var(--font-thai)" }}>
              เกิดข้อผิดพลาด — กรุณาลองอีกครั้ง
            </p>
          )}
        </div>
      )}

      {/* ── Step 3: Results ──────────────────────────────────────── */}
      {step === 3 && data && (
        <div className="animate-fade-in">
          {/* Original score summary */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginBottom: 20,
              padding: "14px 20px",
              borderRadius: "var(--radius-lg)",
              background: "var(--bg-2)",
              border: "1px solid var(--border)",
              flexWrap: "wrap",
            }}
          >
            <span style={{ fontFamily: "var(--font-thai)", fontSize: "0.882rem", color: "var(--text-muted)" }}>
              คะแนนเริ่มต้น:
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "1.4rem",
                fontWeight: 700,
                color: getTier(data.original_risk_class as RiskClass).text,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {Math.round(data.original_risk_score)}
            </span>
            <span
              className="tier-chip"
              style={{
                background: getTier(data.original_risk_class as RiskClass).bg,
                color: getTier(data.original_risk_class as RiskClass).text,
                border: `1.5px solid ${getTier(data.original_risk_class as RiskClass).color}`,
              }}
            >
              {data.original_risk_class}
            </span>
          </div>

          {/* Best intervention banner */}
          {data.best_intervention && (
            <div
              style={{
                padding: "14px 20px",
                borderRadius: "var(--radius-lg)",
                background: "var(--color-risk-low-bg)",
                border: "1.5px solid var(--color-risk-low)",
                marginBottom: 20,
                display: "flex",
                alignItems: "center",
                gap: 10,
                flexWrap: "wrap",
              }}
            >
              <span aria-hidden="true" style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--color-risk-low-text)" }}>★</span>
              <span style={{ fontFamily: "var(--font-thai)", fontSize: "0.94rem", fontWeight: 700, color: "var(--color-risk-low-text)" }}>
                มาตรการแนะนำ:{" "}
                <strong>
                  {INTERVENTIONS.find((iv) => iv.type === data.best_intervention)?.label ?? data.best_intervention}
                </strong>
                {data.best_score_reduction != null && (
                  <span style={{ fontWeight: 500 }}>
                    {" "}(ลดได้ {Math.round(data.best_score_reduction)} คะแนน)
                  </span>
                )}
              </span>
            </div>
          )}

          {/* Scenario cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {data.scenarios.map((s) => {
              const t = getTier(s.new_class as RiskClass);
              const effectivePct = data.original_risk_score > 0
                ? (s.score_reduction / data.original_risk_score) * 100
                : 0;
              const isBest = s.intervention_type === data.best_intervention;
              return (
                <div
                  key={s.intervention_type}
                  style={{
                    background: "var(--bg-2)",
                    border: `1.5px solid ${isBest ? "var(--color-risk-low)" : s.effective ? t.color : "var(--border)"}`,
                    borderRadius: "var(--radius-lg)",
                    padding: "18px 22px",
                    position: "relative",
                    overflow: "hidden",
                  }}
                >
                  {isBest && (
                    <div
                      style={{
                        position: "absolute",
                        top: 0,
                        right: 0,
                        padding: "3px 12px",
                        background: "var(--color-risk-low)",
                        color: "white",
                        fontSize: "0.72rem",
                        fontWeight: 700,
                        fontFamily: "var(--font-thai)",
                        borderBottomLeftRadius: "var(--radius-md)",
                      }}
                    >
                      แนะนำ ★
                    </div>
                  )}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 12,
                      gap: 12,
                      flexWrap: "wrap",
                    }}
                  >
                    <span style={{ fontWeight: 700, fontSize: "0.94rem", fontFamily: "var(--font-thai)", color: "var(--text)" }}>
                      {INTERVENTIONS.find((iv) => iv.type === s.intervention_type)?.label ?? s.intervention_type}
                    </span>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                      <span
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: "1.15rem",
                          fontWeight: 700,
                          color: t.text,
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {Math.round(s.new_score)}
                      </span>
                      <span
                        className="tier-chip"
                        style={{
                          background: t.bg,
                          color: t.text,
                          border: `1px solid ${t.color}`,
                          fontSize: "0.765rem",
                        }}
                      >
                        {s.new_class}
                      </span>
                      <span
                        style={{
                          fontSize: "0.825rem",
                          color: s.score_reduction > 0 ? "var(--color-risk-low-text)" : "var(--text-muted)",
                          fontVariantNumeric: "tabular-nums",
                          fontWeight: 600,
                        }}
                      >
                        {s.score_reduction > 0 ? "↓" : "→"} {Math.round(s.score_reduction)} ({effectivePct.toFixed(0)}%)
                      </span>
                    </div>
                  </div>
                  {/* Score bar */}
                  <div
                    style={{
                      height: 7,
                      borderRadius: 4,
                      background: "var(--bg-3)",
                      overflow: "hidden",
                      marginBottom: 10,
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${Math.min(100, (s.new_score / 100) * 100)}%`,
                        background: t.barColor,
                        transition: "width 0.5s cubic-bezier(0.16, 1, 0.3, 1)",
                        borderRadius: 4,
                      }}
                    />
                  </div>
                  <p style={{ fontSize: "0.855rem", color: "var(--text-muted)", margin: 0, fontFamily: "var(--font-thai)" }}>
                    {s.summary_th}
                  </p>
                </div>
              );
            })}
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 24 }}>
            <button
              onClick={() => setStep(1)}
              className="btn-secondary"
            >
              ← จำลองใหม่
            </button>
            <button
              onClick={() => setStep(1)}
              className="btn-secondary"
            >
              แก้ไขสภาพแวดล้อม
            </button>
            <button
              onClick={() => setStep(2)}
              className="btn-secondary"
            >
              เปลี่ยนมาตรการ
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
