import type { SearchResult } from "../../types";
import { StatusBadge } from "../shared/StatusBadge";
import { VideoPlayer } from "../shared/VideoPlayer";

interface VideoCardProps {
  result: SearchResult;
  rank: number;
}

export function VideoCard({ result, rank }: VideoCardProps) {
  const matchPct = Math.round(result.score * 100);
  const variant =
    matchPct >= 80 ? "green" : matchPct >= 50 ? "yellow" : "gray";

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-surface-card">
      <VideoPlayer src={result.url} startTime={result.timestamp} />
      <div className="flex items-center justify-between px-3 py-2">
        <span className="text-xs text-text-secondary">
          #{rank} &middot; {result.second_formatted}
        </span>
        <StatusBadge label={`${matchPct}%`} variant={variant} />
      </div>
    </div>
  );
}
