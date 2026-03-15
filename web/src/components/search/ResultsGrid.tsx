import type { SearchResult } from "../../types";
import { VideoCard } from "./VideoCard";

interface ResultsGridProps {
  results: SearchResult[];
  columns?: 2 | 4;
}

export function ResultsGrid({ results, columns = 4 }: ResultsGridProps) {
  const gridClass =
    columns === 2
      ? "grid grid-cols-1 gap-4 sm:grid-cols-2"
      : "grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4";

  return (
    <div className={gridClass}>
      {results.map((r, i) => (
        <VideoCard key={`${r.video_id}-${r.frame_index}`} result={r} rank={i + 1} />
      ))}
    </div>
  );
}
