import type { Video } from "../../types";
import { StatusBadge } from "../shared/StatusBadge";

interface VideoLibraryTableProps {
  videos: Video[];
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

export function VideoLibraryTable({ videos }: VideoLibraryTableProps) {
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
          {videos.map((v) => (
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
          ))}
        </tbody>
      </table>
    </div>
  );
}
