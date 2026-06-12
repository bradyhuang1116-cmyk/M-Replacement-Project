"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { ConfigResponse } from "@/types";
import { apiFetch } from "@/lib/api";

export function useConfig() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const fetching = useRef(false);

  const fetchConfig = useCallback(async () => {
    if (fetching.current) return;
    fetching.current = true;
    setLoading(true);
    try {
      const data = await apiFetch<ConfigResponse>("/config");
      setConfig(data);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
      fetching.current = false;
    }
  }, []);

  const saveConfig = useCallback(async (overrides: Record<string, unknown>) => {
    setSaving(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const result = await apiFetch<{ status: string; restart_required: boolean }>(
        "/config",
        {
          method: "PUT",
          body: JSON.stringify({ overrides }),
        }
      );
      setSuccessMsg("Settings saved and applied.");
      return result;
    } catch (e) {
      const msg = (e as Error).message;
      setError(msg);
      throw e;
    } finally {
      setSaving(false);
    }
  }, []);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  return { config, loading, saving, error, successMsg, saveConfig, reload: fetchConfig };
}
