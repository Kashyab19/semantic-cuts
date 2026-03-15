import { useEffect, useRef } from "react";
import type { Video } from "../../types";
import { StatusBadge } from "../shared/StatusBadge";
import { ProgressBar } from "../shared/ProgressBar";
import { useJobProgress } from "../../hooks/useJobProgress";

interface VideoLibraryTableProps {
  videos: Video[];
  onJobComplete?: () => void;
}

function statusVariant(status: string) {
  switch (status) {
    case "completed":
      return "green" as const;
    case "processing":
      return "yellow" as const;
    case "failed":
      return "red" as const;
    default:
      return "gray" as const;
  }
}

function ProcessingRow({ video, onJobComplete }: { video: Video; onJobComplete?: () => void }) {
  const progress = useJobProgress(video.id);
  const firedRef = useRef(false);

  useEffect(() => {
    if (progress.status === "completed" && onJobComplete && !firedRef.current) {
      firedRef.current = true;
      onJobComplete();
    }
  }, [progress.status, onJobComplete]);

  return (
    <tr key={video.id} className="hover:bg-surface-hover">
      <td className="px-3 py-2 font-mono text-xs">{video.id.slice(0, 8)}</td>
      <td className="max-w-xs truncate px-3 py-2">
        <a
          href={video.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent hover:underline"
        >
          {video.url}
        </a>
      </td>
      <td className="px-3 py-2">
        <div className="flex flex-col gap-1">
          <StatusBadge label={video.status} variant={statusVariant(video.status)} />
          {progress.progress !== undefined && (
            <ProgressBar progress={progress.progress} />
          )}
        </div>
      </td>
      <td className="px-3 py-2 text-xs text-text-secondary">{video.created_at}</td>
    </tr>
  );
}

export function VideoLibraryTable({ videos, onJobComplete }: VideoLibraryTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-border text-xs uppercase text-text-secondary">
          <tr>
            <th className="px-3 py-2">Job ID</th>
            <th className="px-3 py-2">Video Source</th>
            <th className="px-3 py-2">Status</th>
            <th className="px-3 py-2">Ingested At</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {videos.map((v) =>
            v.status === "processing" ? (
              <ProcessingRow key={v.id} video={v} onJobComplete={onJobComplete} />
            ) : (
              <tr key={v.id} className="hover:bg-surface-hover">
                <td className="px-3 py-2 font-mono text-xs">{v.id.slice(0, 8)}</td>
                <td className="max-w-xs truncate px-3 py-2">
                  <a
                    href={v.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent hover:underline"
                  >
                    {v.url}
                  </a>
                </td>
                <td className="px-3 py-2">
                  <StatusBadge label={v.status} variant={statusVariant(v.status)} />
                </td>
                <td className="px-3 py-2 text-xs text-text-secondary">{v.created_at}</td>
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  );
}
