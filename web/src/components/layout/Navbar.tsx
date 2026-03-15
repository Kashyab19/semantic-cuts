import { NavLink } from "react-router-dom";
import { useTheme } from "../../hooks/useTheme";

const links = [
  { to: "/search", label: "Search" },
  { to: "/process", label: "Process" },
  { to: "/gallery", label: "Gallery" },
  { to: "/stats", label: "Stats" },
];

function ThemeIcon({ theme }: { theme: "system" | "light" | "dark" }) {
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
  // system
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

export function Navbar() {
  const { theme, toggle } = useTheme();

  return (
    <nav className="border-b border-border bg-surface-card">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-8 px-4">
        <span className="text-sm font-semibold tracking-tight">Semantic Cuts</span>
        <div className="flex gap-1">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-surface-active text-text"
                    : "text-text-secondary hover:bg-surface-hover hover:text-text"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>

        <button
          onClick={toggle}
          className="ml-auto rounded-md p-2 text-text-secondary transition-colors hover:bg-surface-hover hover:text-text"
          title={`Theme: ${theme}`}
        >
          <ThemeIcon theme={theme} />
        </button>
      </div>
    </nav>
  );
}
