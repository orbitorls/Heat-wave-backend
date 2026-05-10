"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { HeatwaveDetectResponse } from "@/lib/api-types";
import { STATIONS_FALLBACK } from "@/lib/stations";
import Link from "next/link";
import { toast } from "@/components/ui/Toast";

// Preset configurations for easy selection
const DETECTION_PRESETS = [
  {
    id: "standard",
    label: "มาตรฐาน",
    desc: "ตรวจจับทั่วไป",
    config: { percentile: 90, minDays: 2, requireWarmNights: false, daysBack: 14 },
  },
  {
    id: "strict",
    label: "เข้มงวด",
    desc: "ตรวจจับเร็วขึ้น",
    config: { percentile: 85, minDays: 2, requireWarmNights: true, daysBack: 14 },
  },
  {
    id: "relaxed",
    label: "ผ่อนปรน",
    desc: "ตรวจจับช้าลง",
    config: { percentile: 95, minDays: 3, requireWarmNights: false, daysBack: 21 },
  },
];

const PERIOD_OPTIONS = [
  { value: 7, label: "7 วัน", desc: "สัปดาห์ล่าสุด" },
  { value: 14, label: "14 วัน", desc: "2 สัปดาห์" },
  { value: 30, label: "30 วัน", desc: "เดือนล่าสุด" },
];

export default function EventsPage() {
  const [stationId, setStationId] = useState("BKK_01");
  const [preset, setPreset] = useState("standard");
  const [daysBack, setDaysBack] = useState(14);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Derived from preset
  const currentPreset = DETECTION_PRESETS.find((p) => p.id === preset)!;
  const { percentile, minDays, requireWarmNights } = currentPreset.config;

  const { mutate, data, isPending, isError, reset } = useMutation<
    HeatwaveDetectResponse,
    Error,
    string
  >({
    mutationFn: (sid) =>
      api.autoDetect(sid, {
        days_back: daysBack,
        baseline_days: 60,
        percentile,
        require_warm_nights: requireWarmNights,
      }),
    onSuccess: (result) => {
      if (result.is_heatwave) {
        toast.warning(`ตรวจพบคลื่นร้อน ${result.consecutive_days} วันต่อเนื่อง`);
      } else {
        toast.success("ไม่พบคลื่นร้อนในช่วงที่วิเคราะห์");
      }
    },
    onError: () => {
      toast.error("ไม่สามารถวิเคราะห์ได้ — อาจไม่มีข้อมูลสถานีนี้");
    },
  });

  function handleSubmit() {
    mutate(stationId);
  }

  return (
    <main id="main-content" style={{ maxWidth: 960, margin: "0 auto", padding: "28px clamp(16px, 4vw, 48px) 80px" }}>
      <Link href="/" className="back-link" style={{ marginBottom: 24 }}>
        ← หน้าหลัก
      </Link>

      <h1 className="text-heading" style={{ color: "var(--text)", marginBottom: 6, fontFamily: "var(--font-thai)", marginTop: 20 }}>
        ตรวจจับคลื่นร้อน
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: 32, fontFamily: "var(--font-thai)", fontSize: "0.94rem" }}>
        วิเคราะห์ข้อมูลจากสถานีตรวจวัดอัตโนมัติ — ไม่ต้องป้อนข้อมูลเอง
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
          gap: 28,
          alignItems: "start",
        }}
      >
        {/* ── Form ──────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Station */}
          <div>
            <label className="field-label">เลือกสถานี</label>
            <select
              value={stationId}
              onChange={(e) => { setStationId(e.target.value); reset(); }}
              className="field-input"
            >
              {STATIONS_FALLBACK.map((s) => (
                <option key={s.station_id} value={s.station_id}>
                  {s.name_th} ({s.station_id})
                </option>
              ))}
            </select>
          </div>

          {/* Detection preset - easy to understand */}
          <div>
            <label className="field-label" style={{ marginBottom: 10, display: "block" }}>
              วิธีตรวจจับ
            </label>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(80px, 1fr))", gap: 10 }}>
              {DETECTION_PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setPreset(p.id)}
                  style={{
                    padding: "14px 12px",
                    borderRadius: "var(--radius-md)",
                    border: `2px solid ${preset === p.id ? "var(--accent)" : "var(--border)"}`,
                    background: preset === p.id ? "var(--accent-bg)" : "var(--bg-2)",
                    color: preset === p.id ? "var(--accent-text)" : "var(--text)",
                    cursor: "pointer",
                    fontFamily: "var(--font-thai)",
                    fontSize: "0.94rem",
                    fontWeight: preset === p.id ? 700 : 500,
                    transition: "all 150ms",
                    minHeight: 60,
                  }}
                >
                  <div style={{ fontWeight: 700 }}>{p.label}</div>
                  <div style={{ fontSize: "0.78rem", opacity: 0.8, marginTop: 2 }}>{p.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Analysis period */}
          <div>
            <label className="field-label" style={{ marginBottom: 10, display: "block" }}>
              ระยะเวลาวิเคราะห์
            </label>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {PERIOD_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setDaysBack(opt.value)}
                  style={{
                    flex: "1 1 80px",
                    padding: "10px 14px",
                    borderRadius: "var(--radius-md)",
                    border: `1.5px solid ${daysBack === opt.value ? "var(--accent)" : "var(--border)"}`,
                    background: daysBack === opt.value ? "var(--accent-bg)" : "var(--bg-2)",
                    color: daysBack === opt.value ? "var(--accent-text)" : "var(--text)",
                    cursor: "pointer",
                    fontFamily: "var(--font-thai)",
                    fontSize: "0.9rem",
                    minHeight: 44,
                  }}
                >
                  <div style={{ fontWeight: daysBack === opt.value ? 700 : 600 }}>{opt.label}</div>
                </button>
              ))}
            </div>
            <p style={{ fontSize: "0.75rem", color: "var(--text-subtle)", marginTop: 6, fontFamily: "var(--font-thai)" }}>
              {PERIOD_OPTIONS.find(o => o.value === daysBack)?.desc}
            </p>
          </div>

          {/* Advanced settings toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            style={{
              padding: "10px 14px",
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              fontSize: "0.85rem",
              fontFamily: "var(--font-thai)",
              cursor: "pointer",
              textAlign: "left",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span style={{ transform: showAdvanced ? "rotate(90deg)" : "rotate(0)", transition: "transform 150ms" }}>▸</span>
            ตั้งค่าขั้นสูง {showAdvanced ? "(ซ่อน)" : "(แสดง)"}
          </button>

          {/* Advanced settings panel */}
          {showAdvanced && (
            <div
              style={{
                padding: "16px",
                background: "var(--bg-2)",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border)",
                fontSize: "0.85rem",
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "8px 16px" }}>
                <div>เปอร์เซ็นไทล์: <strong style={{ color: "var(--text)" }}>{percentile}</strong></div>
                <div>วันต่อเนื่องขั้นต่ำ: <strong style={{ color: "var(--text)" }}>{minDays}</strong></div>
                <div>คืนอบอุ่น: <strong style={{ color: "var(--text)" }}>{requireWarmNights ? "ใช่" : "ไม่"}</strong></div>
                <div>ระยะเวลา: <strong style={{ color: "var(--text)" }}>{daysBack} วัน</strong></div>
              </div>
            </div>
          )}

          <button
            onClick={handleSubmit}
            disabled={isPending}
            className="btn-primary"
          >
            {isPending ? "กำลังวิเคราะห์..." : "ตรวจจับอัตโนมัติ →"}
          </button>

          {isError && (
            <p style={{ fontSize: "0.882rem", color: "var(--color-risk-critical-text)", fontFamily: "var(--font-thai)" }}>
              เกิดข้อผิดพลาด — อาจไม่มีข้อมูลสถานีนี้ ลองเลือกสถานีอื่น
            </p>
          )}
        </div>

        {/* ── Results panel ─────────────────────────────────────── */}
        <div>
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
              เลือกสถานีแล้วกด "ตรวจจับอัตโนมัติ"
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                ดึงข้อมูลย้อนหลังและวิเคราะห์ให้โดยอัตโนมัติ
              </span>
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
              <div className="skeleton" style={{ height: 72, borderRadius: "var(--radius-md)" }} />
              {[70, 50, 85].map((w, i) => (
                <div key={i} className="skeleton" style={{ height: 14, width: `${w}%` }} />
              ))}
              <div className="skeleton" style={{ height: 50, marginTop: 4 }} />
            </div>
          )}

          {data && (
            <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Result banner */}
              <div
                style={{
                  background: data.is_heatwave
                    ? "var(--color-risk-critical-bg)"
                    : "var(--color-risk-low-bg)",
                  border: `2px solid ${data.is_heatwave ? "var(--color-risk-critical)" : "var(--color-risk-low)"}`,
                  borderRadius: "var(--radius-lg)",
                  padding: "20px 24px",
                  display: "flex",
                  alignItems: "center",
                  gap: 18,
                }}
              >
                <div
                  style={{
                    width: 56,
                    height: 56,
                    borderRadius: "50%",
                    background: data.is_heatwave
                      ? "var(--color-risk-critical)"
                      : "var(--color-risk-low)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    fontSize: "1.4rem",
                    fontWeight: 700,
                    color: "white",
                  }}
                  aria-hidden="true"
                >
                  {data.is_heatwave ? "!" : "✓"}
                </div>
                <div>
                  <p
                    style={{
                      fontWeight: 700,
                      fontSize: "1.1rem",
                      color: data.is_heatwave
                        ? "var(--color-risk-critical-text)"
                        : "var(--color-risk-low-text)",
                      margin: 0,
                      fontFamily: "var(--font-thai)",
                      lineHeight: 1.3,
                    }}
                  >
                    {data.is_heatwave ? "ตรวจพบคลื่นร้อน" : "ไม่พบคลื่นร้อน"}
                  </p>
                  {data.event_type && (
                    <p
                      style={{
                        fontSize: "0.855rem",
                        color: data.is_heatwave
                          ? "var(--color-risk-critical-text)"
                          : "var(--color-risk-low-text)",
                        margin: "5px 0 0",
                        fontFamily: "var(--font-thai)",
                        opacity: 0.8,
                      }}
                    >
                      {data.event_type}
                    </p>
                  )}
                </div>
              </div>

              {/* Stats tiles */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(80px, 1fr))", gap: 10 }} className="responsive-stats">
                {[
                  { label: "วันต่อเนื่อง", value: String(data.consecutive_days) },
                  { label: "คืนต่อเนื่อง", value: String(data.consecutive_nights) },
                  { label: "Threshold °C", value: data.percentile_threshold.toFixed(1) },
                ].map(({ label, value }) => (
                  <div
                    key={label}
                    style={{
                      background: "var(--bg-2)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-md)",
                      padding: "12px 14px",
                    }}
                  >
                    <p className="text-label" style={{ color: "var(--text-subtle)", marginBottom: 5 }}>
                      {label}
                    </p>
                    <p
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: "1.3rem",
                        fontWeight: 700,
                        color: "var(--text)",
                        margin: 0,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {value}
                    </p>
                  </div>
                ))}
              </div>

              {/* Trigger reasons */}
              {data.trigger_reason.length > 0 && (
                <div
                  style={{
                    background: "var(--bg-2)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-lg)",
                    padding: "16px 20px",
                  }}
                >
                  <p className="text-label" style={{ color: "var(--text-muted)", marginBottom: 10 }}>
                    สาเหตุที่เกิด
                  </p>
                  <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 7 }}>
                    {data.trigger_reason.map((r, i) => (
                      <li
                        key={i}
                        style={{
                          fontSize: "0.94rem",
                          color: "var(--text)",
                          fontFamily: "var(--font-thai)",
                          paddingLeft: 18,
                          position: "relative",
                          lineHeight: 1.5,
                        }}
                      >
                        <span style={{ position: "absolute", left: 2, color: "var(--accent)", fontWeight: 700 }}>›</span>
                        {r}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Local baseline */}
              {Object.keys(data.local_baseline).length > 0 && (
                <div
                  style={{
                    background: "var(--bg-2)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-lg)",
                    padding: "16px 20px",
                  }}
                >
                  <p className="text-label" style={{ color: "var(--text-muted)", marginBottom: 10 }}>
                    Baseline ท้องถิ่น
                  </p>
                  <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                    {Object.entries(data.local_baseline).map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span
                          style={{
                            fontSize: "0.825rem",
                            color: "var(--text-muted)",
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          {k}
                        </span>
                        <span
                          style={{
                            fontSize: "0.94rem",
                            fontWeight: 700,
                            color: "var(--text)",
                            fontFamily: "var(--font-mono)",
                            fontVariantNumeric: "tabular-nums",
                          }}
                        >
                          {typeof v === "number" ? v.toFixed(2) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
