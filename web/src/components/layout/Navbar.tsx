import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Search" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/admin", label: "Admin" },
];

export function Navbar() {
  return (
    <nav className="border-b border-border bg-surface-card">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-8 px-4">
        <span className="font-serif text-lg font-bold tracking-tight">Semantic Cuts</span>
        <div className="flex gap-1">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-surface-active text-accent-active-text"
                    : "text-text-secondary hover:bg-surface-hover hover:text-text"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
}
