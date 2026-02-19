interface StatusBadgeProps {
  label: string;
  variant?: "green" | "yellow" | "red" | "blue" | "gray";
}

const colors: Record<string, string> = {
  green: "bg-green-100 text-green-700",
  yellow: "bg-yellow-100 text-yellow-700",
  red: "bg-red-100 text-red-700",
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
