import { useCallback, useEffect, useState } from "react";
import { getVideos } from "../api/videos";
import type { Video } from "../types";

export function useVideos(pollInterval?: number) {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await getVideos();
      setVideos(data.videos);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load videos");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    if (pollInterval) {
      const id = setInterval(refresh, pollInterval);
      return () => clearInterval(id);
    }
  }, [refresh, pollInterval]);

  return { videos, loading, error, refresh };
}
