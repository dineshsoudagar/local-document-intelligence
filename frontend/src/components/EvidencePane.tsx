import { useMemo } from "react";
import type { DocumentItem, QuerySource } from "../types";

type EvidencePaneProps = {
  sources: QuerySource[];
  documents: DocumentItem[];
};

export function EvidencePane({ sources, documents }: EvidencePaneProps) {
  const documentNameById = useMemo(
    () =>
      new Map(
        documents.map((document) => [document.doc_id, document.original_filename]),
      ),
    [documents],
  );

  function resolveDisplayName(source: QuerySource) {
    const directName = source.doc_id ? documentNameById.get(source.doc_id) : null;
    if (directName) {
      return directName;
    }

    // If a managed filename accidentally leaks through, recover the doc_id from its stem.
    if (source.original_filename?.startsWith("doc_") && source.original_filename.endsWith(".pdf")) {
      const derivedDocId = source.original_filename.slice(0, -4);
      const derivedName = documentNameById.get(derivedDocId);
      if (derivedName) {
        return derivedName;
      }
    }

    return source.original_filename ?? "Unknown file";
  }

  return (
    <aside className="right-pane">
      <h2>Evidence</h2>

      {/* Empty state before any grounded query has returned evidence. */}
      {sources.length === 0 && <p>No evidence yet</p>}

      <ul className="source-list">
        {sources.map((source) => (
          <li
            key={`${source.doc_id ?? source.original_filename ?? "unknown"}`}
            className="source-item"
          >
            <strong>
              {resolveDisplayName(source)}
              {source.pages.length > 0 ? ` (pages: ${source.pages.join(", ")})` : ""}
            </strong>
          </li>
        ))}
      </ul>
    </aside>
  );
}
