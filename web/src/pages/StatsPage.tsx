import { PageShell } from "../components/layout/PageShell";
import { StatusBadge } from "../components/shared/StatusBadge";
import { Spinner } from "../components/shared/Spinner";
import { useHealth } from "../hooks/useHealth";
import { useVideos } from "../hooks/useVideos";

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

export function StatsPage() {
  const { health, stats, error: healthError } = useHealth();
  const { videos, loading } = useVideos();

  return (
    <PageShell>
      <h1 className="mb-1 text-2xl font-semibold">Stats</h1>
      <p className="mb-6 text-sm text-text-secondary">System health and job overview</p>

      {/* Stat cards */}
      <div className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="rounded-lg border border-border bg-surface-card p-4">
          <p className="text-xs text-text-secondary">Vectors Indexed</p>
          <p className="text-2xl font-bold text-accent">
            {stats?.points_count?.toLocaleString() ?? "—"}
          </p>
        </div>
        <div className="rounded-lg border border-border bg-surface-card p-4">
          <p className="text-xs text-text-secondary">Videos</p>
          <p className="text-2xl font-bold">{videos.length}</p>
        </div>
        <div className="rounded-lg border border-border bg-surface-card p-4">
          <p className="text-xs text-text-secondary">Device</p>
          <p className="text-lg font-bold font-mono">{health?.device ?? "—"}</p>
        </div>
        <div className="rounded-lg border border-border bg-surface-card p-4">
          <p className="text-xs text-text-secondary">Status</p>
          {healthError ? (
            <StatusBadge label="Offline" variant="red" />
          ) : health ? (
            <StatusBadge label="Online" variant="green" />
          ) : (
            <span className="text-xs text-text-tertiary">...</span>
          )}
        </div>
      </div>

      {/* Infrastructure */}
      {health && (
        <div className="mb-8">
          <h2 className="mb-3 text-lg font-semibold">Infrastructure</h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {Object.entries(health.infrastructure).map(([name, status]) => (
              <div
                key={name}
                className="flex items-center justify-between rounded-lg border border-border bg-surface-card p-3"
              >
                <span className="text-sm capitalize">{name}</span>
                <StatusBadge
                  label={status}
                  variant={status === "connected" ? "green" : "red"}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Jobs table */}
      <h2 className="mb-3 text-lg font-semibold">Jobs</h2>
      {loading ? (
        <Spinner />
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border bg-surface-card">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border text-xs uppercase text-text-secondary">
              <tr>
                <th className="px-3 py-2">Job ID</th>
                <th className="px-3 py-2">Source</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Duration</th>
                <th className="px-3 py-2">Created</th>
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
                  <td className="px-3 py-2 text-xs text-text-secondary">
                    {v.duration
                      ? `${Math.floor(v.duration / 60)}m ${Math.round(v.duration % 60)}s`
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-xs text-text-secondary">{v.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PageShell>
  );
}
