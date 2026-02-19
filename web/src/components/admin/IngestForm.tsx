import { useState, type FormEvent } from "react";
import { dispatchVideo } from "../../api/videos";

interface IngestFormProps {
  onDispatched?: () => void;
}

export function IngestForm({ onDispatched }: IngestFormProps) {
  const [url, setUrl] = useState("");
  const [userId, setUserId] = useState("admin");
  const [status, setStatus] = useState<{ type: "success" | "error"; msg: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;

    setSubmitting(true);
    setStatus(null);
    try {
      const res = await dispatchVideo(url.trim(), userId.trim());
      setStatus({ type: "success", msg: `Job queued: ${res.job_id}` });
      setUrl("");
      onDispatched?.();
    } catch (err) {
      setStatus({
        type: "error",
        msg: err instanceof Error ? err.message : "Dispatch failed",
      });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="mb-1 block text-sm font-medium text-text-label">
          Video URL (MP4)
        </label>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/video.mp4"
          className="w-full rounded-lg border border-border px-3 py-2 text-sm focus:border-border-focus focus:ring-2 focus:ring-ring-focus focus:outline-none"
        />
      </div>
      <div>
        <label className="mb-1 block text-sm font-medium text-text-label">
          Uploader ID
        </label>
        <input
          type="text"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          className="w-full rounded-lg border border-border px-3 py-2 text-sm focus:border-border-focus focus:ring-2 focus:ring-ring-focus focus:outline-none"
        />
      </div>
      <button
        type="submit"
        disabled={submitting || !url.trim()}
        className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
      >
        {submitting ? "Dispatching..." : "Dispatch Job"}
      </button>
      {status && (
        <p
          className={`text-sm ${status.type === "success" ? "text-green-600" : "text-red-600"}`}
        >
          {status.msg}
        </p>
      )}
    </form>
  );
}
