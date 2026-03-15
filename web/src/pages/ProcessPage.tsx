import { useState, useCallback } from "react";
import { IngestForm } from "../components/admin/IngestForm";
import { PageShell } from "../components/layout/PageShell";
import { ProgressBar } from "../components/shared/ProgressBar";
import { useJobProgress } from "../hooks/useJobProgress";

function ActiveJobTracker({ jobId, onComplete }: { jobId: string; onComplete: () => void }) {
  const { status, progress } = useJobProgress(jobId);

  if (status === "completed") {
    onComplete();
    return (
      <div className="rounded-lg border border-green-500/30 bg-green-500/10 p-3">
        <p className="text-sm font-medium text-green-600">
          Job {jobId.slice(0, 8)} completed!
        </p>
      </div>
    );
  }

  if (status === "failed") {
    return (
      <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3">
        <p className="text-sm font-medium text-red-600">
          Job {jobId.slice(0, 8)} failed.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-surface-card p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium">Job {jobId.slice(0, 8)}</span>
        <span className="text-xs text-text-secondary">{status ?? "connecting..."}</span>
      </div>
      <ProgressBar progress={progress ?? 0} />
    </div>
  );
}

export function ProcessPage() {
  const [activeJobs, setActiveJobs] = useState<string[]>([]);

  const handleDispatched = useCallback((jobId: string) => {
    setActiveJobs((prev) => [...prev, jobId]);
  }, []);

  const handleJobComplete = useCallback((jobId: string) => {
    setActiveJobs((prev) => prev.filter((id) => id !== jobId));
  }, []);

  return (
    <PageShell>
      <div className="mx-auto max-w-xl">
        <h1 className="mb-1 text-2xl font-semibold">Process Video</h1>
        <p className="mb-6 text-sm text-text-secondary">
          Submit a video URL to extract and index visual moments
        </p>

        <div className="rounded-lg border border-border bg-surface-card p-5">
          <IngestForm onDispatched={handleDispatched} />
        </div>

        {activeJobs.length > 0 && (
          <div className="mt-6 space-y-2">
            <h3 className="text-sm font-semibold text-text-secondary">Active Jobs</h3>
            {activeJobs.map((jobId) => (
              <ActiveJobTracker
                key={jobId}
                jobId={jobId}
                onComplete={() => handleJobComplete(jobId)}
              />
            ))}
          </div>
        )}
      </div>
    </PageShell>
  );
}
