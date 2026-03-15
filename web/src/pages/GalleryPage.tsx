import { useState } from "react";
import { PageShell } from "../components/layout/PageShell";
import { StatusBadge } from "../components/shared/StatusBadge";
import { EmptyState } from "../components/shared/EmptyState";
import { Spinner } from "../components/shared/Spinner";
import { useVideos } from "../hooks/useVideos";
import type { Video } from "../types";

const INFERENCE_URL =
  import.meta.env.VITE_API_INFERENCE_URL ?? "http://localhost:8001";
const PAGE_SIZE = 9;

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

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function VideoCard({ video }: { video: Video }) {
  const [playing, setPlaying] = useState(false);
  const videoSrc = `${INFERENCE_URL}/videos/${video.id}.mp4`;

  return (
    <div className="group overflow-hidden rounded-lg border border-border bg-surface-card">
      {/* Video / Thumbnail area */}
      <div className="relative aspect-video bg-black">
        {playing ? (
          <video
            src={videoSrc}
            controls
            autoPlay
            className="h-full w-full"
          />
        ) : (
          <button
            onClick={() => setPlaying(true)}
            className="flex h-full w-full items-center justify-center transition-colors hover:bg-white/5"
          >
            <div className="flex size-14 items-center justify-center rounded-full bg-accent/90 text-surface shadow-lg transition-transform group-hover:scale-110">
              <svg viewBox="0 0 24 24" fill="currentColor" className="ml-1 size-6">
                <path d="M8 5v14l11-7z" />
              </svg>
            </div>
          </button>
        )}

        {/* Duration badge */}
        {video.duration && !playing && (
          <span className="absolute bottom-2 right-2 rounded bg-black/80 px-1.5 py-0.5 text-xs font-medium text-white">
            {formatDuration(video.duration)}
          </span>
        )}
      </div>

      {/* Info */}
      <div className="p-3">
        <div className="mb-1 flex items-center justify-between">
          <p className="truncate text-sm font-medium">{video.title ?? "Untitled"}</p>
          <StatusBadge label={video.status} variant={statusVariant(video.status)} />
        </div>
        <a
          href={video.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block truncate text-xs text-text-secondary hover:text-accent"
        >
          {video.url}
        </a>
      </div>
    </div>
  );
}

export function GalleryPage() {
  const { videos, loading, error } = useVideos();
  const [page, setPage] = useState(1);

  const completed = videos.filter((v) => v.status === "completed");
  const totalPages = Math.max(1, Math.ceil(completed.length / PAGE_SIZE));
  const pageVideos = completed.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <PageShell>
      <h1 className="mb-1 text-2xl font-semibold">Gallery</h1>
      <p className="mb-6 text-sm text-text-secondary">
        {completed.length} video{completed.length !== 1 ? "s" : ""} ready to watch
      </p>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {loading && <Spinner />}
      {!loading && completed.length === 0 && !error && (
        <EmptyState message="No videos yet. Head to Process to add one." />
      )}

      {!loading && completed.length > 0 && (
        <>
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {pageVideos.map((v) => (
              <VideoCard key={v.id} video={v} />
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-8 flex items-center justify-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-40"
              >
                Prev
              </button>
              {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                    p === page
                      ? "bg-accent text-surface"
                      : "border border-border hover:bg-surface-hover"
                  }`}
                >
                  {p}
                </button>
              ))}
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-40"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </PageShell>
  );
}
