import { IngestForm } from "../components/admin/IngestForm";
import { VideoLibraryTable } from "../components/admin/VideoLibraryTable";
import { PageShell } from "../components/layout/PageShell";
import { EmptyState } from "../components/shared/EmptyState";
import { Spinner } from "../components/shared/Spinner";
import { useVideos } from "../hooks/useVideos";

export function AdminPage() {
  const { videos, loading, error, refresh } = useVideos(15_000);

  return (
    <PageShell>
      <h1 className="mb-6 font-serif text-2xl font-bold">Admin Console</h1>
      <p className="mb-6 text-sm text-text-secondary">Content Ingestion Pipeline</p>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        <div className="rounded-xl border border-border bg-surface-card p-5">
          <h2 className="mb-4 font-serif text-lg font-semibold">Ingest Video</h2>
          <IngestForm onDispatched={refresh} />
        </div>

        <div className="lg:col-span-2">
          <h2 className="mb-4 font-serif text-lg font-semibold">Video Library</h2>
          {error && <p className="text-sm text-red-600">{error}</p>}
          {loading && <Spinner />}
          {!loading && videos.length === 0 && !error && (
            <EmptyState message="No videos found. Upload one to start." />
          )}
          {!loading && videos.length > 0 && (
            <div className="rounded-xl border border-border bg-surface-card">
              <VideoLibraryTable videos={videos} />
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
