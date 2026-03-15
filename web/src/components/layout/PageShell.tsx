import type { ReactNode } from "react";

interface PageShellProps {
  children: ReactNode;
}

export function PageShell({ children }: PageShellProps) {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      {children}
    </div>
  );
}
