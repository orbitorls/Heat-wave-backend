"use client";

import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ActionCardRequest } from "@/lib/api-types";
import { toast } from "@/components/ui/Toast";

export function useActionCard() {
  return useMutation({
    mutationFn: (req: ActionCardRequest) => api.actionCard(req),
    onSuccess: () => {
      toast.success("Action Card ถูกสร้างแล้ว พร้อมสำหรับการพิมพ์");
    },
    onError: () => {
      toast.error("ไม่สามารถสร้าง Action Card ได้ — ลองอีกครั้ง");
    },
  });
}
