const API_BASE = "http://localhost:8000/api/v1";

export const FALLBACK_API_KEY = "dev-api-key";

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
  const res = await fetch(`${API_BASE}${path}`, {
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
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: authHeaders(init?.headers),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.blob();
}


export function apiSSE(path: string): EventSource {
  return new EventSource(`${API_BASE}${path}`);
}

export const API_BASE_URL = API_BASE;
