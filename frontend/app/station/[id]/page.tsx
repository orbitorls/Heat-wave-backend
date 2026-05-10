"use client";

import { use, useState } from "react";
import Link from "next/link";
import { useForecast } from "@/hooks/useForecast";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { LowConfidenceBadge } from "@/components/forecast/LowConfidenceBadge";
import { TierBadge } from "@/components/ui/TierBadge";
import { Stat } from "@/components/ui/Stat";
import { STATIONS_FALLBACK } from "@/lib/stations";
import { hiToRiskClass, getTier } from "@/lib/tier";
import { formatHI, formatElevation, formatHorizon } from "@/lib/format";

type Tab = "overview" | "forecast" | "risk";
const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "ภาพรวม" },
  { id: "forecast", label: "พยากรณ์" },
  { id: "risk", label: "ความเสี่ยง" },
];

export default function StationPage(props: { params: Promise<{ id: string }> }) {
  const { id } = use(props.params);
  const [tab, setTab] = useState<Tab>("overview");

  const station = STATIONS_FALLBACK.find((s) => s.station_id === id);
  const { data: forecast, isLoading, isError } = useForecast(id, [6, 12, 24, 48, 72]);

  if (!station) {
    return (
      <div style={{ padding: "64px clamp(16px,4vw,48px)", textAlign: "center" }}>
        <p style={{ color: "var(--text-muted)", fontFamily: "var(--font-thai)", marginBottom: 20 }}>
          ไม่พบสถานี: <strong>{id}</strong>
        </p>
        <Link href="/" className="back-link">
          กลับหน้าหลัก
        </Link>
      </div>
    );
  }

  const primaryPoint = forecast?.forecasts?.[0];
  const hiC = primaryPoint?.heat_index_c;
  const riskClass = hiC != null ? hiToRiskClass(hiC) : null;
  const tier = riskClass ? getTier(riskClass) : null;
  const isCritical = riskClass === "Critical";

  return (
    <div
      style={{
        maxWidth: 960,
        margin: "0 auto",
        padding: "0 clamp(16px, 3vw, 32px) 64px",
      }}
    >
      {/* Breadcrumb */}
      <nav
        aria-label="เส้นทาง"
        style={{ padding: "16px 0", display: "flex", alignItems: "center", gap: 6 }}
      >
        <Link href="/" className="back-link" style={{ minHeight: 32, padding: "5px 10px" }}>
          หน้าหลัก
        </Link>
        <span style={{ color: "var(--border-strong)", fontSize: "0.9rem" }} aria-hidden="true">/</span>
        <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontFamily: "var(--font-thai)" }}>
          {station.name_th}
        </span>
      </nav>

      {/* Critical tier — full-bleed danger header */}
      {isCritical && tier && (
        <div
          role="alert"
          style={{
            background: tier.bg,
            padding: "12px 20px",
            borderRadius: "var(--radius-md)",
            marginBottom: 16,
            display: "flex",
            alignItems: "center",
            gap: 12,
          }}
        >
          <span style={{ color: tier.color, fontSize: "1.2rem" }} aria-hidden="true">⚠</span>
          <p style={{ margin: 0, fontFamily: "var(--font-thai)", fontWeight: 700, color: tier.text }}>
            ระดับวิกฤต — ความร้อนเป็นอันตรายต่อสุขภาพ หยุดกิจกรรมกลางแจ้ง
          </p>
        </div>
      )}

      {/* Station header — data card */}
      <header
        style={{
          padding: "16px 0 20px",
          borderBottom: "1px solid var(--border)",
          marginBottom: 0,
        }}
      >
        <p className="text-label" style={{ color: "var(--text-subtle)", marginBottom: 8 }}>
          {station.station_id} · {formatElevation(station.elevation_m)} MSL
        </p>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div>
            <h1
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "clamp(1.4rem, 3vw, 1.9rem)",
                fontWeight: 700,
                color: "var(--text)",
                margin: "0 0 12px",
                lineHeight: 1.2,
                letterSpacing: "-0.02em",
              }}
            >
              {station.name_th}
            </h1>

            {/* Heat index + tier */}
            {isLoading && <div className="skeleton" style={{ height: 42, width: 140 }} />}
            {!isLoading && hiC != null && riskClass && tier && (
              <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
                <span className="text-data-xl" style={{ color: tier.text, fontSize: "2.6rem" }}>
                  {formatHI(hiC)}
                </span>
                <TierBadge tier={riskClass} size="lg" showDot />
                {forecast?.low_confidence && (
                  <LowConfidenceBadge reason={forecast.confidence_reason} />
                )}
              </div>
            )}
          </div>

          {/* Next horizons row */}
          {!isLoading && !isError && forecast && (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {forecast.forecasts.slice(0, 3).map((pt) => {
                const t = getTier(hiToRiskClass(pt.heat_index_c));
                return (
                  <div
                    key={pt.horizon_h}
                    style={{
                      background: t.bg,
                      borderRadius: "var(--radius-md)",
                      padding: "10px 14px",
                      textAlign: "center",
                      minWidth: 68,
                    }}
                  >
                    <p style={{ margin: 0, fontSize: "0.7rem", fontWeight: 700, color: t.text, letterSpacing: "0.03em", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
                      +{pt.horizon_h}h
                    </p>
                    <p className="text-data-md" style={{ margin: "4px 0 0", color: t.text }}>
                      {formatHI(pt.heat_index_c)}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </header>

      {/* Tab bar with proper ARIA */}
      <div
        role="tablist"
        aria-label="ข้อมูลสถานี"
        style={{
          display: "flex",
          gap: 4,
          padding: "14px 0",
          borderBottom: "1px solid var(--border)",
          marginBottom: 24,
        }}
      >
        {TABS.map(({ id: t, label }) => (
          <button
            key={t}
            role="tab"
            id={`tab-${t}`}
            aria-selected={tab === t}
            aria-controls={`tabpanel-${t}`}
            onClick={() => setTab(t)}
            style={{
              padding: "8px 18px",
              minHeight: 38,
              fontSize: "0.9rem",
              fontWeight: tab === t ? 700 : 500,
              color: tab === t ? "white" : "var(--text-muted)",
              background: tab === t ? "var(--accent)" : "transparent",
              border: `1px solid ${tab === t ? "var(--accent)" : "var(--border)"}`,
              borderRadius: "var(--radius-md)",
              cursor: "pointer",
              fontFamily: "var(--font-thai)",
              transition: "all 150ms",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Overview ─────────────────────────────────────────────── */}
      <div
        id="tabpanel-overview"
        role="tabpanel"
        aria-labelledby="tab-overview"
        hidden={tab !== "overview"}
        className="animate-fade-in"
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
            gap: 10,
            marginBottom: 24,
          }}
        >
          {[
            { label: "ละติจูด", value: station.lat.toFixed(4) },
            { label: "ลองจิจูด", value: station.lon.toFixed(4) },
            { label: "ความสูง", value: formatElevation(station.elevation_m) },
            { label: "รุ่นโมเดล", value: forecast?.model_version ?? (isLoading ? "…" : "—") },
          ].map(({ label, value }) => (
            <div
              key={label}
              style={{
                background: "var(--bg-2)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                padding: "14px 16px",
              }}
            >
              <Stat label={label} value={value} size="sm" />
            </div>
          ))}
        </div>

        {/* Forecast preview */}
        {isLoading && (
          <div style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "20px 24px" }}>
            <div className="skeleton" style={{ height: 18, width: 160, marginBottom: 16 }} />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="skeleton" style={{ height: 72, borderRadius: "var(--radius-md)" }} />
              ))}
            </div>
          </div>
        )}

        {isError && !isLoading && (
          <div
            style={{
              padding: "20px 24px",
              borderRadius: "var(--radius-lg)",
              background: "var(--bg-2)",
              border: "1px solid var(--border)",
              color: "var(--text-muted)",
              fontFamily: "var(--font-thai)",
              fontSize: "0.94rem",
            }}
          >
            ไม่สามารถโหลดข้อมูลพยากรณ์ได้ — ต้องการข้อมูลย้อนหลัง (parquet) เพิ่มเติม
          </div>
        )}

        {!isLoading && !isError && forecast && (
          <div style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "20px 24px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 12 }}>
              <h2 style={{ fontSize: "0.95rem", fontWeight: 700, margin: 0, fontFamily: "var(--font-thai)", color: "var(--text)" }}>
                พยากรณ์ล่วงหน้า
              </h2>
              <button
                onClick={() => setTab("forecast")}
                style={{ fontSize: "0.85rem", color: "var(--accent)", background: "none", border: "none", cursor: "pointer", fontFamily: "var(--font-thai)", fontWeight: 700, padding: "4px 0", minHeight: "auto", minWidth: "auto", display: "inline-flex", alignItems: "center", gap: 4 }}
              >
                ดูกราฟ
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                  <path d="M2 6h8M6.5 2.5 10 6l-3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(90px, 1fr))", gap: 8 }}>
              {forecast.forecasts.map((pt) => {
                const t = getTier(hiToRiskClass(pt.heat_index_c));
                return (
                  <div key={pt.horizon_h} style={{ background: t.bg, borderRadius: "var(--radius-md)", padding: "12px 8px", textAlign: "center" }}>
                    <p style={{ margin: 0, fontSize: "0.7rem", color: t.text, fontWeight: 700, marginBottom: 6, fontFamily: "var(--font-mono)", letterSpacing: "0.02em" }}>
                      +{formatHorizon(pt.horizon_h)}
                    </p>
                    <p className="text-data-md" style={{ margin: 0, color: t.text }}>
                      {formatHI(pt.heat_index_c)}
                    </p>
                    <p style={{ margin: "4px 0 0", fontSize: "0.7rem", color: t.text, fontFamily: "var(--font-thai)", fontWeight: 600 }}>
                      {t.label}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── Forecast tab ─────────────────────────────────────────── */}
      <div
        id="tabpanel-forecast"
        role="tabpanel"
        aria-labelledby="tab-forecast"
        hidden={tab !== "forecast"}
        className="animate-fade-in"
      >
        {isLoading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="skeleton" style={{ height: 24, width: 240 }} />
            <div className="skeleton" style={{ height: 220, borderRadius: "var(--radius-lg)" }} />
          </div>
        )}
        {isError && !isLoading && (
          <div style={{ padding: "20px 24px", borderRadius: "var(--radius-lg)", background: "var(--bg-2)", border: "1px solid var(--border)", color: "var(--text-muted)", fontFamily: "var(--font-thai)" }}>
            ไม่สามารถโหลดข้อมูล forecast ได้
          </div>
        )}
        {forecast && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 12 }}>
              <h2 style={{ fontSize: "1rem", fontWeight: 700, margin: 0, fontFamily: "var(--font-thai)", color: "var(--text)" }}>
                พยากรณ์ (ช่วงความเชื่อมั่น 90%)
              </h2>
              {forecast.low_confidence && <LowConfidenceBadge reason={forecast.confidence_reason} />}
            </div>
            <div style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "20px 24px", marginBottom: 12 }}>
              <ForecastChart forecasts={forecast.forecasts} lowConfidence={forecast.low_confidence} />
            </div>
            <p style={{ fontSize: "0.75rem", color: "var(--text-subtle)", fontFamily: "var(--font-mono)" }}>
              รุ่น: {forecast.model_version} · {new Date(forecast.generated_at).toLocaleString("th-TH")}
            </p>
          </div>
        )}
      </div>

      {/* ── Risk tab ─────────────────────────────────────────────── */}
      <div
        id="tabpanel-risk"
        role="tabpanel"
        aria-labelledby="tab-risk"
        hidden={tab !== "risk"}
        className="animate-fade-in"
      >
        {riskClass && tier && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Current HI risk summary */}
            <div
              style={{
                background: tier.bg,
                padding: "20px 24px",
                borderRadius: "var(--radius-lg)",
                display: "flex",
                alignItems: "center",
                gap: 20,
                flexWrap: "wrap",
              }}
            >
              <div>
                <p className="text-label" style={{ color: tier.text, marginBottom: 6, opacity: 0.75 }}>
                  ดัชนีความร้อนปัจจุบัน
                </p>
                <span className="text-data-xl" style={{ color: tier.text }}>
                  {formatHI(hiC!)}
                </span>
              </div>
              <div>
                <p className="text-label" style={{ color: tier.text, marginBottom: 6, opacity: 0.75 }}>
                  ระดับความเสี่ยง
                </p>
                <TierBadge tier={riskClass} size="lg" showDot />
              </div>
            </div>

            {/* Forecast risk overview */}
            {forecast && (
              <div style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "20px 24px" }}>
                <h2 style={{ margin: "0 0 16px", fontSize: "0.95rem", fontWeight: 700, color: "var(--text)", fontFamily: "var(--font-thai)" }}>
                  แนวโน้มความเสี่ยงในอีก {formatHorizon(forecast.forecasts.at(-1)?.horizon_h ?? 72)}
                </h2>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {forecast.forecasts.map((pt) => {
                    const rc = hiToRiskClass(pt.heat_index_c);
                    const t = getTier(rc);
                    return (
                      <div
                        key={pt.horizon_h}
                        style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, minWidth: 56 }}
                      >
                        <p style={{ margin: 0, fontSize: "0.68rem", fontFamily: "var(--font-mono)", color: "var(--text-subtle)", letterSpacing: "0.02em" }}>
                          +{pt.horizon_h}h
                        </p>
                        <div
                          style={{
                            width: 36,
                            height: 36,
                            borderRadius: "var(--radius-sm)",
                            background: t.bg,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            border: `1px solid ${t.color}44`,
                          }}
                        >
                          <span className="text-data-sm" style={{ color: t.text, fontSize: "0.78rem" }}>
                            {formatHI(pt.heat_index_c)}
                          </span>
                        </div>
                        <p style={{ margin: 0, fontSize: "0.65rem", color: t.text, fontWeight: 700, fontFamily: "var(--font-thai)" }}>
                          {t.label}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Link to detailed risk tool */}
            <Link
              href="/tools/risk"
              className="btn-secondary"
              style={{ width: "fit-content" }}
            >
              เช็คความเสี่ยงเฉพาะบุคคล
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <path d="M3 7h8M7.5 3 11 7l-3.5 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </Link>
          </div>
        )}

        {!riskClass && !isLoading && (
          <div style={{ color: "var(--text-muted)", fontFamily: "var(--font-thai)", fontSize: "0.94rem" }}>
            ยังไม่มีข้อมูล forecast — ต้องการข้อมูลย้อนหลัง (parquet) เพิ่มเติม
          </div>
        )}

        {isLoading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="skeleton" style={{ height: 100, borderRadius: "var(--radius-lg)" }} />
            <div className="skeleton" style={{ height: 140, borderRadius: "var(--radius-lg)" }} />
          </div>
        )}
      </div>
    </div>
  );
}
