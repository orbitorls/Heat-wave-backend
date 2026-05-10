"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useForecast(stationId: string, horizons: number[] = [6, 24, 48]) {
  return useQuery({
    queryKey: ["forecast", stationId, horizons],
    queryFn: () => api.forecast({ station_id: stationId, horizons }),
    staleTime: 60_000,
    enabled: !!stationId,
    retry: 1,
  });
}
