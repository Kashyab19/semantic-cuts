import { useState, type FormEvent } from "react";
import { dispatchVideo } from "../api/videos";
import { PageShell } from "../components/layout/PageShell";
import { ResultsGrid } from "../components/search/ResultsGrid";
import { EmptyState } from "../components/shared/EmptyState";
import { Spinner } from "../components/shared/Spinner";
import { StatusBadge } from "../components/shared/StatusBadge";
import { useHealth } from "../hooks/useHealth";
import { useSearch } from "../hooks/useSearch";

export function DashboardPage() {
  const { query, results, loading, error, search } = useSearch(4);
  const { health, stats, error: healthError } = useHealth();

  // Sidebar ingest form state
  const [videoUrl, setVideoUrl] = useState("");
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleIngest(e: FormEvent) {
    e.preventDefault();
    if (!videoUrl.trim()) return;
    setSubmitting(true);
    setIngestMsg(null);
    try {
      const res = await dispatchVideo(videoUrl.trim(), "dashboard_user");
      setIngestMsg(`Job queued: ${res.job_id}`);
      setVideoUrl("");
    } catch (err) {
      setIngestMsg(err instanceof Error ? err.message : "Failed");
    } finally {
      setSubmitting(false);
    }
  }

  // Search bar state
  const [input, setInput] = useState("");
  function handleSearch(e: FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (trimmed) search(trimmed);
  }

  return (
    <PageShell>
      <h1 className="mb-6 text-2xl font-semibold">Dashboard</h1>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-4">
        {/* Sidebar */}
        <aside className="space-y-6">
          {/* Add Video */}
          <div className="rounded-lg border border-border bg-surface-card p-4">
            <h3 className="mb-3 text-sm font-semibold">Add Video</h3>
            <form onSubmit={handleIngest} className="space-y-2">
              <input
                type="text"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                placeholder="https://example.com/video.mp4"
                className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm focus:border-border-focus focus:ring-2 focus:ring-ring-focus focus:outline-none"
              />
              <button
                type="submit"
                disabled={submitting || !videoUrl.trim()}
                className="w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-surface hover:bg-accent-hover disabled:opacity-50"
              >
                {submitting ? "Indexing..." : "Index Video"}
              </button>
            </form>
            {ingestMsg && (
              <p className="mt-2 text-xs text-green-600">{ingestMsg}</p>
            )}
          </div>

          {/* System Status */}
          <div className="rounded-lg border border-border bg-surface-card p-4">
            <h3 className="mb-3 text-sm font-semibold">System Status</h3>
            {healthError ? (
              <StatusBadge label="Offline" variant="red" />
            ) : health ? (
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Qdrant</span>
                  <StatusBadge
                    label={health.infrastructure.qdrant}
                    variant={health.infrastructure.qdrant === "connected" ? "green" : "red"}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Redpanda</span>
                  <StatusBadge
                    label={health.infrastructure.redpanda}
                    variant={health.infrastructure.redpanda === "connected" ? "green" : "red"}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Redis</span>
                  <StatusBadge
                    label={health.infrastructure.redis}
                    variant={health.infrastructure.redis === "connected" ? "green" : "red"}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Device</span>
                  <span className="font-mono">{health.device}</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-text-tertiary">Loading...</p>
            )}
          </div>

          {/* Indexed Vectors */}
          {stats && (
            <div className="rounded-lg border border-border bg-surface-card p-4">
              <h3 className="mb-1 text-sm font-semibold">Indexed Vectors</h3>
              <p className="text-2xl font-bold text-accent">
                {stats.points_count?.toLocaleString() ?? "—"}
              </p>
            </div>
          )}
        </aside>

        {/* Main content */}
        <div className="lg:col-span-3">
          <form onSubmit={handleSearch} className="mb-6 flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe a moment..."
              className="flex-1 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm focus:border-border-focus focus:ring-2 focus:ring-ring-focus focus:outline-none"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="rounded-lg bg-accent px-6 py-2.5 text-sm font-medium text-surface hover:bg-accent-hover disabled:opacity-50"
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </form>

          {error && <p className="text-sm text-red-600">{error}</p>}
          {loading && <Spinner />}
          {!loading && query && results.length === 0 && !error && (
            <EmptyState message="No matches found." />
          )}
          {!loading && results.length > 0 && (
            <>
              <h2 className="mb-4 text-lg font-semibold">Top Matches</h2>
              <ResultsGrid results={results} columns={2} />
            </>
          )}
        </div>
      </div>
    </PageShell>
  );
}
