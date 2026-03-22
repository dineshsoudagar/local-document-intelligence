import { useEffect, useRef, useState } from "react";
import type { DocumentItem } from "../types";

type DocumentsPaneProps = {
  documents: DocumentItem[];
  selectedDocId: string | null;
  isUploading: boolean;
  uploadError: string | null;
  error: string | null;
  deletingDocId: string | null;
  openMenuDocId: string | null;
  onSelectDocument: (docId: string) => void;
  onDeleteDocument: (document: DocumentItem) => Promise<void>;
  onToggleMenu: (docId: string | null) => void;
  onUploadFile: (file: File) => Promise<void>;
};

export function DocumentsPane({
  documents,
  selectedDocId,
  isUploading,
  uploadError,
  error,
  deletingDocId,
  openMenuDocId,
  onSelectDocument,
  onDeleteDocument,
  onToggleMenu,
  onUploadFile,
}: DocumentsPaneProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Ignore non-file drags so text selections and other drag events do not trigger upload UI.
  function isFileDrag(event: DragEvent | React.DragEvent<HTMLElement>) {
    return Array.from(event.dataTransfer?.types ?? []).includes("Files");
  }

  // Prefer the dragged file item, then fall back to the browser's file list.
  function getDraggedFile(event: React.DragEvent<HTMLElement>) {
    const item = Array.from(event.dataTransfer.items ?? []).find(
      (candidate) => candidate.kind === "file",
    );

    if (item) {
      return item.getAsFile();
    }

    return event.dataTransfer.files?.[0] ?? null;
  }

  useEffect(() => {
    // Prevent the browser from navigating away when a file is dropped outside the pane.
    function preventWindowDrop(event: DragEvent) {
      if (!isFileDrag(event)) {
        return;
      }

      event.preventDefault();
    }

    window.addEventListener("dragover", preventWindowDrop);
    window.addEventListener("drop", preventWindowDrop);

    return () => {
      window.removeEventListener("dragover", preventWindowDrop);
      window.removeEventListener("drop", preventWindowDrop);
    };
  }, []);

  // Opens the hidden native file picker.
  function handleOpenFilePicker() {
    fileInputRef.current?.click();
  }

  // Accept one chosen file and reuse the same upload flow as drag and drop.
  async function handleFileInputChange(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    await onUploadFile(file);
    event.target.value = "";
  }

  // Prevent browser default so dropping a file stays inside the app.
  function handleDragOver(event: React.DragEvent<HTMLElement>) {
    if (!isFileDrag(event)) {
      return;
    }

    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsDragOver(true);
  }

  // Remove drag highlight only when the pointer fully leaves the drop zone.
  function handleDragLeave(event: React.DragEvent<HTMLElement>) {
    if (!isFileDrag(event)) {
      return;
    }

    event.preventDefault();

    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }

    setIsDragOver(false);
  }

  // Accept one dropped file and send it through the same upload path.
  async function handleDrop(event: React.DragEvent<HTMLElement>) {
    if (!isFileDrag(event)) {
      return;
    }

    event.preventDefault();
    setIsDragOver(false);

    const file = getDraggedFile(event);
    if (!file) {
      return;
    }

    await onUploadFile(file);
  }

  return (
    <aside
      className={isDragOver ? "left-pane drag-over" : "left-pane"}
      onDragEnter={handleDragOver}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <h2>Documents</h2>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        style={{ display: "none" }}
        onChange={handleFileInputChange}
      />

      {/* The visible upload button controls the hidden file input above. */}
      <div className="upload-dropzone">
        <button
          type="button"
          className="submit-button"
          onClick={handleOpenFilePicker}
          disabled={isUploading}
        >
          {isUploading ? "Uploading..." : "Upload PDF"}
        </button>

        <p>Drag and drop a PDF here, or use the upload button.</p>
      </div>

      {uploadError && <p>{uploadError}</p>}
      {error && <p>{error}</p>}

      <ul className="document-list">
        {documents.map((document) => (
          <li key={document.doc_id}>
            <div className="document-row">
              {/* Selecting a document makes it available for single-document queries. */}
              <button
                type="button"
                className={
                  document.doc_id === selectedDocId
                    ? "document-button selected"
                    : "document-button"
                }
                onClick={() => onSelectDocument(document.doc_id)}
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
