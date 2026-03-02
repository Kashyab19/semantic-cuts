import { useState, type FormEvent } from "react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading?: boolean;
  placeholder?: string;
}

export function SearchBar({
  onSearch,
  loading,
  placeholder = "Describe a moment (e.g. 'a red car', 'people laughing')...",
}: SearchBarProps) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (trimmed) onSearch(trimmed);
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder}
        className="flex-1 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm focus:border-border-focus focus:ring-2 focus:ring-ring-focus focus:outline-none"
      />
      <button
        type="submit"
        disabled={loading || !value.trim()}
        className="rounded-lg bg-accent px-6 py-2.5 text-sm font-medium text-surface hover:bg-accent-hover disabled:opacity-50"
      >
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
}
