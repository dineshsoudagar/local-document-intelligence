import type { DocumentItem } from "../types";

type DocumentsPaneProps = {
  documents: DocumentItem[];
  isFetchingDocuments: boolean;
  selectedDocIds: string[];
  error: string | null;
  deletingDocId: string | null;
  openMenuDocId: string | null;
  onSelectDocument: (docIds: string[]) => void;
  onToggleSelectAll: () => void;
  onDeleteDocument: (document: DocumentItem) => Promise<void>;
  onToggleMenu: (docId: string | null) => void;
};

export function DocumentsPane({
  documents,
  isFetchingDocuments,
  selectedDocIds,
  error,
  deletingDocId,
  openMenuDocId,
  onSelectDocument,
  onToggleSelectAll,
  onDeleteDocument,
  onToggleMenu,
}: DocumentsPaneProps) {
  const hasSelection = selectedDocIds.length > 0;

  return (
    <aside className="left-pane">
      <div className="documents-header">
        <h2>Documents</h2>

        <button
          type="button"
          className="documents-clear-button"
          onClick={onToggleSelectAll}
          disabled={documents.length === 0}
        >
          {hasSelection ? "Unselect All" : "Select All"}
        </button>
      </div>
      {error && <p>{error}</p>}
      {isFetchingDocuments && <p className="documents-status">Fetching documents...</p>}

      <ul className="document-list">
        {documents.map((document) => (
          <li key={document.doc_id}>
            <div className="document-row">
              {/* Document selection is toggle-based so the user can build a
                  filtered subset of documents directly from the list. */}
              <button
                type="button"
                className={
                  selectedDocIds.includes(document.doc_id)
                    ? "document-button selected"
                    : "document-button"
                }
                onClick={() => {
                  if (selectedDocIds.includes(document.doc_id)) {
                    onSelectDocument(
                      selectedDocIds.filter((docId) => docId !== document.doc_id),
                    );
                    return;
                  }

                  onSelectDocument([...selectedDocIds, document.doc_id]);
                }}
              >
                {document.original_filename}
              </button>

              <div
                className="document-actions"
                onClick={(event) => event.stopPropagation()}
              >
                {/* Keep menu clicks local so the global window click handler does not close it immediately. */}
                <button
                  type="button"
                  className="document-menu-button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onToggleMenu(
                      openMenuDocId === document.doc_id ? null : document.doc_id,
                    );
                  }}
                  disabled={deletingDocId === document.doc_id}
                  aria-label={`Open actions for ${document.original_filename}`}
                  aria-haspopup="menu"
                  aria-expanded={openMenuDocId === document.doc_id}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="5" cy="12" r="1.75" />
                    <circle cx="12" cy="12" r="1.75" />
                    <circle cx="19" cy="12" r="1.75" />
                  </svg>
                </button>

                {openMenuDocId === document.doc_id && (
                  // Render actions only for the document whose menu is currently open.
                  <div className="document-menu" role="menu">
                    <button
                      type="button"
                      className="document-menu-item delete"
                      onClick={(event) => {
                        event.stopPropagation();
                        void onDeleteDocument(document);
                      }}
                      disabled={deletingDocId === document.doc_id}
                      role="menuitem"
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M9 3h6l1 2h4v2H4V5h4l1-2Z" />
                        <path d="M6 9h12l-1 10a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L6 9Z" />
                        <path d="M10 11v6" />
                        <path d="M14 11v6" />
                      </svg>
                      <span>
                        {deletingDocId === document.doc_id ? "Deleting..." : "Delete"}
                      </span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}
