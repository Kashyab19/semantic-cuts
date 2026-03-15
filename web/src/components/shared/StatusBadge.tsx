interface StatusBadgeProps {
  label: string;
  variant?: "green" | "yellow" | "red" | "blue" | "gray";
}

const colors: Record<string, string> = {
  green: "bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-400",
  yellow: "bg-yellow-50 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-400",
  red: "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-400",
  blue: "bg-surface-active text-accent-active-text",
  gray: "bg-surface-hover text-text-secondary",
};

export function StatusBadge({ label, variant = "gray" }: StatusBadgeProps) {
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${colors[variant]}`}
    >
      {label}
    </span>
  );
}
