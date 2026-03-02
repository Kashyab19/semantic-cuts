import { useCallback, useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";

const STORAGE_KEY = "theme";

function getSystemPreference(): "light" | "dark" {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function applyTheme(theme: Theme) {
  const resolved = theme === "system" ? getSystemPreference() : theme;
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    return (stored as Theme) ?? "system";
  });

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  // Listen for system preference changes when in "system" mode
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    function handler() {
      if (theme === "system") applyTheme("system");
    }
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [theme]);

  const toggle = useCallback(() => {
    setThemeState((prev) => {
      const next: Theme =
        prev === "system" ? "light" : prev === "light" ? "dark" : "system";
      localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  }, []);

  return { theme, toggle } as const;
}
