import type { QuerySource } from "../types";

type EvidencePaneProps = {
  sources: QuerySource[];
};

export function EvidencePane({ sources }: EvidencePaneProps) {
  // Sort evidence by rank so the most relevant chunks always appear first.
  const sortedSources = [...sources].sort((left, right) => left.rank - right.rank);

  return (
    <aside className="right-pane">
      <h2>Evidence</h2>

      {/* Empty state before any grounded query has returned evidence. */}
      {sortedSources.length === 0 && <p>No evidence yet</p>}

      <ul className="source-list">
        {sortedSources.map((source) => (
          <li key={source.chunk_id} className="source-item">
            {/* Prefer the original uploaded filename, then fall back to backend source metadata. */}
            <div>
              <strong>
                {source.original_filename ?? source.source_file ?? "Unknown file"}
              </strong>
            </div>

            <div>
              Pages: {source.page_start ?? "-"} - {source.page_end ?? "-"}
            </div>

            <div>
              Headings:{" "}
              {source.headings && source.headings.length > 0
                ? source.headings.join(" > ")
                : "-"}
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}
