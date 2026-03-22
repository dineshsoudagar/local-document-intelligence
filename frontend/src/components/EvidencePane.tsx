import type { QuerySource } from "../types";

type EvidencePaneProps = {
  sources: QuerySource[];
};

export function EvidencePane({ sources }: EvidencePaneProps) {
  const sortedSources = [...sources].sort((left, right) => left.rank - right.rank);

  return (
    <aside className="right-pane">
      <h2>Evidence</h2>

      {sortedSources.length === 0 && <p>No evidence yet</p>}

      <ul className="source-list">
        {sortedSources.map((source) => (
          <li key={source.chunk_id} className="source-item">
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
