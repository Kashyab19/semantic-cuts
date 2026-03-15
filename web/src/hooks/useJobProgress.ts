import { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_ORCHESTRATOR_URL ?? "http://localhost:8000";

export interface JobProgress {
  status: string;
  pending: number;
  total: number;
  progress: number;
}

export function useJobProgress(jobId: string | null) {
  const [data, setData] = useState<JobProgress | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const es = new EventSource(`${API_URL}/video/${jobId}/progress`);
    esRef.current = es;

    es.onopen = () => setIsConnected(true);

    es.onmessage = (event) => {
      const parsed: JobProgress = JSON.parse(event.data);
      setData(parsed);

      if (parsed.status === "completed" || parsed.status === "failed") {
        es.close();
        setIsConnected(false);
      }
    };

    es.onerror = () => {
      setIsConnected(false);
      es.close();
    };

    return () => {
      es.close();
      setIsConnected(false);
    };
  }, [jobId]);

  return { ...data, isConnected };
}
