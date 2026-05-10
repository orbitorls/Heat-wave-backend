"use client";

import Link from "next/link";
import type { Station, ForecastResponse } from "@/lib/api-types";
import { hiToRiskClass, getTier } from "@/lib/tier";
import { formatHI, formatHorizon } from "@/lib/format";
import { LowConfidenceBadge } from "@/components/forecast/LowConfidenceBadge";
import { TierBadge } from "@/components/ui/TierBadge";
import { TrendSpark } from "@/components/charts/TrendSpark";

interface StationCardProps {
  station: Station;
  forecast?: ForecastResponse;
  forecastLoading?: boolean;
  forecastError?: boolean;
  animationDelay?: number;
}

export function StationCard({
  station,
  forecast,
  forecastLoading,
  forecastError,
  animationDelay = 0,
}: StationCardProps) {
  const primaryPoint = forecast?.forecasts?.[0];
  const hiC = primaryPoint?.heat_index_c;
  const riskClass = hiC != null ? hiToRiskClass(hiC) : null;
  const tier = riskClass ? getTier(riskClass) : null;

  return (
    <Link
      href={`/station/${station.station_id}`}
      style={{ textDecoration: "none", display: "block" }}
      aria-label={`ดูข้อมูลสถานี ${station.name_th}`}
    >
      <article
        className="animate-fade-up card-hover"
        style={{
          animationDelay: `${animationDelay}ms`,
          background: "var(--bg-2)",
          border: `1.5px solid ${tier ? tier.color + "55" : "var(--border)"}`,
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          cursor: "pointer",
          position: "relative",
          container: "card / inline-size",
        }}
      >
        {/* Critical tier: full-bleed danger stripe at top */}
        {riskClass === "Critical" && (
          <div
            style={{
              height: 3,
              background: tier!.color,
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
            }}
            aria-hidden="true"
          />
        )}

        <div style={{ padding: "20px 24px 0", paddingTop: riskClass === "Critical" ? 23 : 20 }}>
          {/* Station ID + name */}
          <header style={{ marginBottom: 16 }}>
            <p
              className="text-label"
              style={{ color: "var(--text-subtle)", marginBottom: 4, fontSize: "0.7rem" }}
            >
              {station.station_id}
            </p>
            <h3
              style={{
                fontSize: "1.05rem",
                fontWeight: 700,
                color: "var(--text)",
                margin: 0,
                fontFamily: "var(--font-thai)",
                lineHeight: 1.3,
                letterSpacing: "-0.01em",
              }}
            >
              {station.name_th}
            </h3>
          </header>

          {/* Loading */}
          {forecastLoading && (
            <div style={{ marginBottom: 20 }}>
              <div className="skeleton" style={{ height: 44, width: 120, marginBottom: 8 }} />
              <div className="skeleton" style={{ height: 16, width: 72 }} />
            </div>
          )}

          {/* Error */}
          {forecastError && !forecastLoading && (
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", fontFamily: "var(--font-thai)", marginBottom: 16 }}>
              ไม่สามารถโหลดข้อมูลได้
            </p>
          )}

          {/* Heat index + tier + spark */}
          {!forecastLoading && !forecastError && hiC != null && tier && riskClass && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 8 }}>
                <div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                    <span
                      className="text-data-xl"
                      style={{ color: tier.text }}
                    >
                      {formatHI(hiC)}
                    </span>
                    <TierBadge tier={riskClass} size="sm" />
                  </div>
                  <p
                    style={{
                      fontSize: "0.78rem",
                      color: "var(--text-subtle)",
                      marginTop: 4,
                      fontFamily: "var(--font-thai)",
                    }}
                  >
                    {formatHorizon(primaryPoint!.horizon_h)} ข้างหน้า
                  </p>
                </div>

                {forecast && forecast.forecasts.length > 1 && (
                  <TrendSpark points={forecast.forecasts} width={72} height={32} />
                )}
              </div>
            </div>
          )}

          {/* Mini forecast strip */}
          {!forecastLoading && !forecastError && forecast && forecast.forecasts.length > 1 && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 6,
                marginBottom: 12,
              }}
            >
              {forecast.forecasts.slice(0, 3).map((pt) => {
                const t = getTier(hiToRiskClass(pt.heat_index_c));
                return (
                  <div
                    key={pt.horizon_h}
                    style={{
                      background: t.bg,
                      borderRadius: "var(--radius-sm)",
                      padding: "8px 6px",
                      textAlign: "center",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.68rem",
                        color: t.text,
                        marginBottom: 3,
                        fontWeight: 700,
                        letterSpacing: "0.02em",
                        textTransform: "uppercase",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      +{pt.horizon_h}h
                    </div>
                    <div className="text-data-sm" style={{ color: t.text }}>
                      {formatHI(pt.heat_index_c)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {forecast?.low_confidence && (
            <div style={{ marginBottom: 12 }}>
              <LowConfidenceBadge reason={forecast.confidence_reason} />
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: "12px 24px",
            background: tier ? `${tier.color}0e` : "transparent",
            borderTop: "1px solid var(--border)",
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "center",
          }}
        >
          <span
            style={{
              fontSize: "0.78rem",
              color: tier ? tier.text : "var(--text-subtle)",
              fontWeight: 600,
              fontFamily: "var(--font-thai)",
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            ดูข้อมูล
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M2 6h8M6.5 2.5 10 6l-3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </span>
        </div>
      </article>
    </Link>
  );
}
