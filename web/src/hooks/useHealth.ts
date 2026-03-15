import { useEffect, useState } from "react";
import { getHealth, getStats } from "../api/health";
import type { InferenceHealth, StatsResponse } from "../types";

export function useHealth(pollInterval: number = 10_000) {
  const [health, setHealth] = useState<InferenceHealth | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function poll() {
      try {
        const [h, s] = await Promise.all([getHealth(), getStats()]);
        setHealth(h);
        setStats(s);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Health check failed");
      }
    }

    poll();
    const id = setInterval(poll, pollInterval);
    return () => clearInterval(id);
  }, [pollInterval]);

  return { health, stats, error };
}
