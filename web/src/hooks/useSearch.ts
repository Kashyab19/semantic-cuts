import { useCallback, useState } from "react";
import { searchVideos } from "../api/search";
import type { SearchResult } from "../types";

export function useSearch(limit: number = 8) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(
    async (q: string) => {
      setQuery(q);
      setLoading(true);
      setError(null);
      try {
        const data = await searchVideos(q, limit);
        setResults(data.results);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Search failed");
        setResults([]);
      } finally {
        setLoading(false);
      }
    },
    [limit]
  );

  return { query, results, loading, error, search };
}
