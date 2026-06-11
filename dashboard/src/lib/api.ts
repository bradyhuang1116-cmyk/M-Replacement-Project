const API_BASE = "http://localhost:8000/api/v1";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

export function apiSSE(path: string): EventSource {
  return new EventSource(`${API_BASE}${path}`);
}

export const API_BASE_URL = API_BASE;
