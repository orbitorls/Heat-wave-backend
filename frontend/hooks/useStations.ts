"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { STATIONS_FALLBACK } from "@/lib/stations";

export function useStations() {
  return useQuery({
    queryKey: ["stations"],
    queryFn: api.stations,
    staleTime: Infinity, // station list is static
    placeholderData: STATIONS_FALLBACK,
  });
}
