import { Link } from "react-router-dom";

/* ---------- inline SVG icons ---------- */
function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="size-8 text-accent">
      <circle cx={11} cy={11} r={8} />
      <path d="m21 21-4.35-4.35" strokeLinecap="round" />
    </svg>
  );
}

function SceneIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="size-8 text-accent">
      <rect x={2} y={3} width={20} height={14} rx={2} />
      <path d="m8 21 4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M7 8h2M15 8h2" strokeLinecap="round" />
    </svg>
  );
}

function DistributedIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="size-8 text-accent">
      <rect x={1} y={4} width={8} height={6} rx={1} />
      <rect x={15} y={4} width={8} height={6} rx={1} />
      <rect x={8} y={14} width={8} height={6} rx={1} />
      <path d="M5 10v2a2 2 0 0 0 2 2h3M19 10v2a2 2 0 0 1-2 2h-3" strokeLinecap="round" />
    </svg>
  );
}

/* ---------- feature data ---------- */
const features = [
  {
    icon: <SearchIcon />,
    title: "Semantic Search",
    description:
      "Search your videos using natural language. Find the exact moment you're looking for without scrubbing through hours of footage.",
  },
  {
    icon: <SceneIcon />,
    title: "Scene Detection",
    description:
      "Automatically detects scene boundaries using histogram analysis so only key frames are processed — fast and efficient.",
  },
  {
    icon: <DistributedIcon />,
    title: "Distributed Processing",
    description:
      "Processes videos with a containerised pipeline using Celery workers, so indexing scales with your library.",
  },
];

export function LandingPage() {
  return (
    <>
      {/* ---- Navbar ---- */}
      <nav className="border-b border-border bg-surface-card">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <span className="font-serif text-lg font-bold tracking-tight">Semantic Cuts</span>
          <Link
            to="/playground"
            className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
          >
            Open Playground
          </Link>
        </div>
      </nav>

      {/* ---- Hero ---- */}
      <section className="mx-auto max-w-4xl px-4 py-24 text-center">
        <h1 className="font-serif text-5xl font-bold leading-tight tracking-tight text-text sm:text-6xl">
          Find any moment
          <br />
          in any video.
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-text-secondary">
          Semantic Cuts indexes your video library and lets you search it with plain
          English. Powered by CLIP embeddings and scene-aware frame extraction.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link
            to="/playground"
            className="rounded-md bg-accent px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
          >
            Try the Playground
          </Link>
          <a
            href="https://github.com/Kashyab19/semantic-cuts"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md border border-border px-6 py-2.5 text-sm font-medium text-text transition-colors hover:bg-surface-hover"
          >
            View on GitHub
          </a>
        </div>
      </section>

      {/* ---- Features ---- */}
      <section className="mx-auto max-w-6xl px-4 pb-24">
        <h2 className="text-center font-serif text-3xl font-bold tracking-tight text-text">
          How it works
        </h2>
        <div className="mt-12 grid gap-6 sm:grid-cols-3">
          {features.map((f) => (
            <div key={f.title} className="rounded-lg border border-border bg-surface-card p-6">
              <div className="mb-4">{f.icon}</div>
              <h3 className="font-serif text-lg font-semibold text-text">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-text-secondary">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- Bottom CTA ---- */}
      <section className="border-t border-border bg-surface-card py-16 text-center">
        <h2 className="font-serif text-2xl font-bold text-text">Ready to search your videos?</h2>
        <Link
          to="/playground"
          className="mt-6 inline-block rounded-md bg-accent px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
        >
          Try the Playground
        </Link>
      </section>

      {/* ---- Footer ---- */}
      <footer className="py-8 text-center text-sm text-text-tertiary">
        Semantic Cuts — open-source video search
      </footer>
    </>
  );
}
