"use client";

import { useStations } from "@/hooks/useStations";
import { useForecast } from "@/hooks/useForecast";
import { StationCard } from "./StationCard";

function StationWithForecast({
  stationId,
  index,
}: {
  stationId: string;
  index: number;
}) {
  const { data: allStations } = useStations();
  const station = allStations?.find((s) => s.station_id === stationId);
  const {
    data: forecast,
    isLoading,
    isError,
  } = useForecast(stationId, [6, 24, 48]);

  if (!station) return null;

  return (
    <StationCard
      station={station}
      forecast={forecast}
      forecastLoading={isLoading}
      forecastError={isError}
      animationDelay={index * 40}
    />
  );
}

function SkeletonCard() {
  return (
    <div
      style={{
        background: "var(--bg-2)",
        border: "1.5px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        overflow: "hidden",
      }}
    >
      <div style={{ height: 4, background: "var(--bg-3)" }} />
      <div style={{ padding: "20px 22px 16px" }}>
        <div className="skeleton" style={{ height: 12, width: 60, marginBottom: 8 }} />
        <div className="skeleton" style={{ height: 20, width: 140, marginBottom: 20 }} />
        <div className="skeleton" style={{ height: 44, width: 110, marginBottom: 8 }} />
        <div className="skeleton" style={{ height: 14, width: 90 }} />
      </div>
      <div
        style={{
          padding: "12px 22px",
          borderTop: "1px solid var(--border)",
          background: "var(--bg)",
          display: "flex",
          justifyContent: "flex-end",
        }}
      >
        <div className="skeleton" style={{ height: 14, width: 80 }} />
      </div>
    </div>
  );
}

export function StationGrid() {
  const { data: stations, isLoading } = useStations();

  return (
    <section aria-label="5 สถานีตรวจวัดอากาศ">
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(290px, 1fr))",
          gap: 18,
        }}
      >
        {isLoading
          ? Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)
          : (stations ?? []).map((station, i) => (
              <StationWithForecast
                key={station.station_id}
                stationId={station.station_id}
                index={i}
              />
            ))}
      </div>
    </section>
  );
}
