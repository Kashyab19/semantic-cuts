import { PageShell } from "../components/layout/PageShell";
import { ResultsGrid } from "../components/search/ResultsGrid";
import { SearchBar } from "../components/search/SearchBar";
import { EmptyState } from "../components/shared/EmptyState";
import { Spinner } from "../components/shared/Spinner";
import { useSearch } from "../hooks/useSearch";

export function SearchPage() {
  const { query, results, loading, error, search } = useSearch(8);

  return (
    <PageShell>
      <div className="mb-8">
        <h1 className="mb-1 text-2xl font-semibold">Semantic Cuts</h1>
        <p className="text-sm text-text-secondary">Multimodal Video Search Engine</p>
      </div>

      <SearchBar onSearch={search} loading={loading} />

      <div className="mt-6">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {loading && <Spinner />}
        {!loading && query && results.length === 0 && !error && (
          <EmptyState message="No matching moments found." />
        )}
        {!loading && results.length > 0 && (
          <>
            <p className="mb-4 text-sm text-text-secondary">
              Found {results.length} relevant moments
            </p>
            <ResultsGrid results={results} columns={4} />
          </>
        )}
      </div>
    </PageShell>
  );
}
