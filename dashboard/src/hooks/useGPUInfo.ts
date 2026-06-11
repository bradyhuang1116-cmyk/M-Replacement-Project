"use client";

import { useState, useEffect, useRef } from "react";
import type { GpuMetrics } from "@/types";
import { apiFetch } from "@/lib/api";

export function useGPUInfo(intervalMs = 2000) {
  const [data, setData] = useState<GpuMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetching = useRef(false);

  useEffect(() => {
    let mounted = true;

    async function poll() {
      if (fetching.current) return;
      fetching.current = true;
      try {
        const gpu = await apiFetch<GpuMetrics>("/gpu");
        if (mounted && !("error" in gpu)) {
          setData(gpu);
          setError(null);
        }
      } catch (e) {
        if (mounted) setError((e as Error).message);
      } finally {
        fetching.current = false;
      }
    }

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { data, error };
}
