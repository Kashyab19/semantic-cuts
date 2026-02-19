const ORCHESTRATOR_URL =
  import.meta.env.VITE_API_ORCHESTRATOR_URL ?? "http://localhost:8000";
const INFERENCE_URL =
  import.meta.env.VITE_API_INFERENCE_URL ?? "http://localhost:8001";

async function request<T>(base: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export function orchestrator<T>(path: string, init?: RequestInit) {
  return request<T>(ORCHESTRATOR_URL, path, init);
}

export function inference<T>(path: string, init?: RequestInit) {
  return request<T>(INFERENCE_URL, path, init);
}
