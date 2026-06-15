const API_PATH_PREFIX = "/api/v1";

export const FALLBACK_API_KEY = "dev-api-key";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function resolveApiOrigin(): string {
  const configuredBase = process.env.NEXT_PUBLIC_API_BASE?.trim();
  if (configuredBase) return trimTrailingSlash(configuredBase);

  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:8000`;
  }

  return "http://localhost:8000";
}

export function getApiBaseUrl(): string {
  return `${resolveApiOrigin()}${API_PATH_PREFIX}`;
}

function getApiKey(): string {
  if (typeof window === "undefined") return FALLBACK_API_KEY;
  return window.localStorage.getItem("internalApiKey") || FALLBACK_API_KEY;
}

function authHeaders(extra?: HeadersInit): HeadersInit {
  return {
    Authorization: `Bearer ${getApiKey()}`,
    ...extra,
  };
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...init?.headers,
    },
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function apiFetchBlob(path: string, init?: RequestInit): Promise<Blob> {
  const res = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: authHeaders(init?.headers),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.blob();
}


export function apiSSE(path: string): EventSource {
  return new EventSource(`${getApiBaseUrl()}${path}`);
}

export const API_BASE_URL = getApiBaseUrl;

// ── 推理模式（本地 / 云端）──
export type InferenceMode = "cloud" | "local";

export async function getMode(): Promise<InferenceMode> {
  const r = await apiFetch<{ mode: InferenceMode }>("/mode");
  return r.mode;
}

export async function setMode(mode: InferenceMode): Promise<InferenceMode> {
  const r = await apiFetch<{ mode: InferenceMode }>("/mode", {
    method: "PUT",
    body: JSON.stringify({ mode }),
  });
  return r.mode;
}
