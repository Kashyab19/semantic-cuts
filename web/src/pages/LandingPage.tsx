import { Link } from "react-router-dom";
import { useTheme } from "../hooks/useTheme";

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

function ThemeToggleIcon({ theme }: { theme: "system" | "light" | "dark" }) {
  if (theme === "dark") {
    return (
      <svg viewBox="0 0 20 20" fill="currentColor" className="size-4">
        <path
          fillRule="evenodd"
          d="M7.455 2.004a.75.75 0 01.26.77 7 7 0 009.958 7.967.75.75 0 011.067.853A8.5 8.5 0 116.93 1.837a.75.75 0 01.526.167z"
          clipRule="evenodd"
        />
      </svg>
    );
  }
  if (theme === "light") {
    return (
      <svg viewBox="0 0 20 20" fill="currentColor" className="size-4">
        <path d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zm0 13a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zm5-5a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5h-1.5A.75.75 0 0115 10zM2 10a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5h-1.5A.75.75 0 012 10zm12.07-4.07a.75.75 0 010 1.06l-1.06 1.06a.75.75 0 11-1.06-1.06l1.06-1.06a.75.75 0 011.06 0zM8.05 13.95a.75.75 0 010 1.06l-1.06 1.06a.75.75 0 01-1.06-1.06l1.06-1.06a.75.75 0 011.06 0zM5.93 5.93a.75.75 0 010 1.06L4.87 8.05a.75.75 0 01-1.06-1.06l1.06-1.06a.75.75 0 011.06 0zm9.1 9.1a.75.75 0 010 1.06l-1.06 1.06a.75.75 0 11-1.06-1.06l1.06-1.06a.75.75 0 011.06 0z" />
        <path
          fillRule="evenodd"
          d="M10 4a6 6 0 100 12 6 6 0 000-12zm-4 6a4 4 0 118 0 4 4 0 01-8 0z"
          clipRule="evenodd"
        />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className="size-4">
      <path
        fillRule="evenodd"
        d="M2 4.25A2.25 2.25 0 014.25 2h11.5A2.25 2.25 0 0118 4.25v8.5A2.25 2.25 0 0115.75 15h-3.105a3.501 3.501 0 001.1 1.677A.75.75 0 0113.26 18H6.74a.75.75 0 01-.484-1.323A3.501 3.501 0 007.355 15H4.25A2.25 2.25 0 012 12.75v-8.5zm1.5 0a.75.75 0 01.75-.75h11.5a.75.75 0 01.75.75v7.5a.75.75 0 01-.75.75H4.25a.75.75 0 01-.75-.75v-7.5z"
        clipRule="evenodd"
      />
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
  const { theme, toggle } = useTheme();

  return (
    <>
      {/* ---- Navbar ---- */}
      <nav className="border-b border-border bg-surface-card">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <span className="text-sm font-semibold tracking-tight">Semantic Cuts</span>
          <div className="flex items-center gap-3">
            <button
              onClick={toggle}
              className="rounded-md p-2 text-text-secondary transition-colors hover:bg-surface-hover hover:text-text"
              title={`Theme: ${theme}`}
            >
              <ThemeToggleIcon theme={theme} />
            </button>
            <Link
              to="/playground"
              className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-surface transition-colors hover:bg-accent-hover"
            >
              Open Playground
            </Link>
          </div>
        </div>
      </nav>

      {/* ---- Hero ---- */}
      <section className="mx-auto max-w-4xl px-4 py-28 text-center">
        <h1 className="text-4xl font-semibold leading-tight tracking-tight text-text sm:text-5xl">
          Find any moment
          <br />
          in any video.
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-base text-text-secondary">
          Semantic Cuts indexes your video library and lets you search it with plain
          English. Powered by CLIP embeddings and scene-aware frame extraction.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link
            to="/playground"
            className="rounded-md bg-accent px-6 py-2.5 text-sm font-medium text-surface transition-colors hover:bg-accent-hover"
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
      <section className="mx-auto max-w-6xl px-4 pb-28">
        <h2 className="text-center text-2xl font-semibold tracking-tight text-text">
          How it works
        </h2>
        <div className="mt-12 grid gap-6 sm:grid-cols-3">
          {features.map((f) => (
            <div key={f.title} className="rounded-lg border border-border bg-surface-card p-6">
              <div className="mb-4">{f.icon}</div>
              <h3 className="text-base font-semibold text-text">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-text-secondary">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- Bottom CTA ---- */}
      <section className="border-t border-border bg-surface-card py-16 text-center">
        <h2 className="text-xl font-semibold text-text">Ready to search your videos?</h2>
        <Link
          to="/playground"
          className="mt-6 inline-block rounded-md bg-accent px-6 py-2.5 text-sm font-medium text-surface transition-colors hover:bg-accent-hover"
        >
          Try the Playground
        </Link>
      </section>

      {/* ---- Footer ---- */}
      <footer className="py-8 text-center text-sm text-text-tertiary">
        <p>&copy; 2026 <a href="https://kashyab.xyz" target="_blank" rel="noopener noreferrer" className="text-text-secondary hover:text-text transition-colors">Kashyab Murali</a></p>
        <p className="mt-1">
          <a href="mailto:hi@kashyab.xyz" className="hover:text-text transition-colors">hi@kashyab.xyz</a>
        </p>
      </footer>
    </>
  );
}
