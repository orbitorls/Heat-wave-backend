"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { RiskScoreRequest } from "@/lib/api-types";

export function useRiskScore(req: RiskScoreRequest | null) {
  return useQuery({
    queryKey: ["risk", req],
    queryFn: () => api.riskScore(req!),
    enabled: req !== null,
    staleTime: 30_000,
  });
}
