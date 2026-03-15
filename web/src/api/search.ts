import type { SearchResponse } from "../types";
import { inference } from "./client";

export function searchVideos(query: string, limit: number = 8) {
  return inference<SearchResponse>(
    `/search?query=${encodeURIComponent(query)}&limit=${limit}`
  );
}
