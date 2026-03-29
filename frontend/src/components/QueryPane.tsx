import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage, DocumentItem, QuerySource, QueryStatus, UiQueryMode } from "../types";
import { MarkdownText } from "./MarkdownText";

type QueryPaneProps = {
  documents: DocumentItem[];
  messages: ChatMessage[];
  theme: "light" | "dark";
  uiMode: UiQueryMode;
  queryStatus: QueryStatus;
  queryError: string | null;
  queryText: string;
  isSubmitting: boolean;
  isUploading: boolean;
  uploadError: string | null;
  onModeChange: (mode: UiQueryMode) => void;
  onQueryTextChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  onUploadFile: (file: File) => Promise<void>;
};

export function QueryPane({
  documents,
  messages,
  theme,
  uiMode,
  queryStatus,
  queryError,
  queryText,
  isSubmitting,
  isUploading,
  uploadError,
  onModeChange,
  onQueryTextChange,
  onSubmit,
  onStop,
  onToggleTheme,
  onOpenSettings,
  onUploadFile,
}: QueryPaneProps) {
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const documentNameById = useMemo(
    () =>
      new Map(
        documents.map((document) => [document.doc_id, document.original_filename]),
      ),
    [documents],
  );

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, queryStatus]);

  useEffect(() => {
    function preventWindowDrop(event: DragEvent) {
      if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) {
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

  function isFileDrag(event: DragEvent | React.DragEvent<HTMLElement>) {
    return Array.from(event.dataTransfer?.types ?? []).includes("Files");
  }

  function getDraggedFile(event: React.DragEvent<HTMLElement>) {
    const item = Array.from(event.dataTransfer.items ?? []).find(
      (candidate) => candidate.kind === "file",
    );

    if (item) {
      return item.getAsFile();
    }

    return event.dataTransfer.files?.[0] ?? null;
  }

  function handleOpenFilePicker() {
    fileInputRef.current?.click();
  }

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

  function handleDragOver(event: React.DragEvent<HTMLElement>) {
    if (!isFileDrag(event)) {
      return;
    }

    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsDragOver(true);
  }

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

  function resolveDisplayName(source: QuerySource) {
    const directName = source.doc_id ? documentNameById.get(source.doc_id) : null;
    if (directName) {
      return directName;
    }

    if (source.original_filename?.startsWith("doc_") && source.original_filename.endsWith(".pdf")) {
      const derivedDocId = source.original_filename.slice(0, -4);
      const derivedName = documentNameById.get(derivedDocId);
      if (derivedName) {
        return derivedName;
      }
    }

    return source.original_filename ?? "Unknown file";
  }

  function formatSources(sources: QuerySource[]) {
    return sources
      .map((source) => {
        const displayName = resolveDisplayName(source);
        if (source.pages.length > 0) {
          return `${displayName} • pages ${source.pages.join(", ")}`;
        }
        return displayName;
      })
      .join("  |  ");
  }

  return (
    <main className="center-pane">
      <div className="center-header">
        <div className="center-header-top">
          <div className="center-header-copy">
            <h2>Local Document Assistant</h2>
            <p className="center-header-subtitle">
              Grounded answers over your local documents
            </p>
          </div>

          <div className="center-header-actions">
            <button
              type="button"
              className="header-action-button settings-button"
              onClick={onOpenSettings}
              aria-label="Open setup settings"
              title="Open setup settings"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 8.75A3.25 3.25 0 1 0 12 15.25A3.25 3.25 0 1 0 12 8.75Z" />
                <path d="M19.4 15A1 1 0 0 0 19.6 16.1L20.1 16.95A1 1 0 0 1 19.75 18.3L18.3 19.75A1 1 0 0 1 16.95 20.1L16.1 19.6A1 1 0 0 0 15 19.4L14.05 19.8A1 1 0 0 0 13.45 20.72V21.75A1 1 0 0 1 12.45 22.75H11.55A1 1 0 0 1 10.55 21.75V20.72A1 1 0 0 0 9.95 19.8L9 19.4A1 1 0 0 0 7.9 19.6L7.05 20.1A1 1 0 0 1 5.7 19.75L4.25 18.3A1 1 0 0 1 3.9 16.95L4.4 16.1A1 1 0 0 0 4.6 15L4.2 14.05A1 1 0 0 0 3.28 13.45H2.25A1 1 0 0 1 1.25 12.45V11.55A1 1 0 0 1 2.25 10.55H3.28A1 1 0 0 0 4.2 9.95L4.6 9A1 1 0 0 0 4.4 7.9L3.9 7.05A1 1 0 0 1 4.25 5.7L5.7 4.25A1 1 0 0 1 7.05 3.9L7.9 4.4A1 1 0 0 0 9 4.6L9.95 4.2A1 1 0 0 0 10.55 3.28V2.25A1 1 0 0 1 11.55 1.25H12.45A1 1 0 0 1 13.45 2.25V3.28A1 1 0 0 0 14.05 4.2L15 4.6A1 1 0 0 0 16.1 4.4L16.95 3.9A1 1 0 0 1 18.3 4.25L19.75 5.7A1 1 0 0 1 20.1 7.05L19.6 7.9A1 1 0 0 0 19.4 9L19.8 9.95A1 1 0 0 0 20.72 10.55H21.75A1 1 0 0 1 22.75 11.55V12.45A1 1 0 0 1 21.75 13.45H20.72A1 1 0 0 0 19.8 14.05L19.4 15Z" />
              </svg>
              <span>Settings</span>
            </button>

            <button
              type="button"
              className="header-action-button theme-toggle"
              onClick={onToggleTheme}
              aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
              title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            >
              {theme === "light" ? (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2.5" />
                  <path d="M12 19.5V22" />
                  <path d="M4.93 4.93l1.77 1.77" />
                  <path d="M17.3 17.3l1.77 1.77" />
                  <path d="M2 12h2.5" />
                  <path d="M19.5 12H22" />
                  <path d="M4.93 19.07l1.77-1.77" />
                  <path d="M17.3 6.7l1.77-1.77" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="chat-window">
        {messages.length === 0 ? (
          <div className="chat-empty-state">
            <h3>Start a conversation</h3>
            <p>Send a message to see the chat transcript appear here.</p>
          </div>
        ) : (
          <div className="chat-transcript" aria-live="polite">
            {messages.map((message) => (
              <div
                key={message.id}
                className={
                  message.role === "user"
                    ? "chat-row chat-row-user"
                    : "chat-row chat-row-assistant"
                }
              >
                <div className="chat-message-block">
                  <div
                    className={
                      message.role === "user"
                        ? "chat-bubble chat-bubble-user"
                        : "chat-bubble chat-bubble-assistant"
                    }
                  >
                    <div className="chat-bubble-label">
                      {message.role === "user" ? "You" : "Assistant"}
                    </div>
                    <div className="chat-bubble-text">
                      {message.content ? (
                        <MarkdownText content={message.content} />
                      ) : (
                        message.status === "streaming" ? "Thinking..." : ""
                      )}
                    </div>
                  </div>
                  {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                    <div className="chat-bubble-sources">
                      {formatSources(message.sources)}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={transcriptEndRef} />
          </div>
        )}
      </div>

      <div className="query-controls">
        <div className="query-controls-row">
          <div className="query-controls-label">Select mode :</div>

          <div className="scope-toggle">
            <button
              type="button"
              className={uiMode === "auto" ? "scope-button selected" : "scope-button"}
              onClick={() => onModeChange("auto")}
            >
              Auto
            </button>

            <button
              type="button"
              className={uiMode === "chat" ? "scope-button selected" : "scope-button"}
              onClick={() => onModeChange("chat")}
            >
              Chat
            </button>

            <button
              type="button"
              className={uiMode === "corpus" ? "scope-button selected" : "scope-button"}
              onClick={() => onModeChange("corpus")}
            >
              Corpus
            </button>

            <button
              type="button"
              className={uiMode === "document" ? "scope-button selected" : "scope-button"}
              onClick={() => onModeChange("document")}
            >
              Selected Documents
            </button>
          </div>
        </div>

        {queryError && <p className="query-error">{queryError}</p>}
      </div>

      <div
        className={isDragOver ? "composer-section drag-over" : "composer-section"}
        onDragEnter={handleDragOver}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          style={{ display: "none" }}
          onChange={handleFileInputChange}
        />

        <label className="composer-label" htmlFor="query-input">
          Message
        </label>

        <p className="composer-helper">
          Drag and drop a PDF anywhere in this box, or use Upload PDF.
        </p>

        {uploadError && <p className="query-error">{uploadError}</p>}

        <div className="composer-layout">
          <textarea
            id="query-input"
            className="query-input"
            placeholder="Type your message or query your documents"
            value={queryText}
            rows={3}
            onChange={(event) => onQueryTextChange(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends the message; Shift+Enter inserts a newline.
              if (event.key === "Enter" && !event.shiftKey && !isSubmitting) {
                event.preventDefault();
                onSubmit();
              }
            }}
          />

          <div className="composer-actions">
            <div className="composer-actions-top">
              <button
                type="button"
                className={
                  isSubmitting
                    ? "submit-button stop-button"
                    : "submit-button ask-button"
                }
                onClick={isSubmitting ? onStop : onSubmit}
              >
                {isSubmitting ? "Stop" : "Send"}
              </button>
            </div>

            <button
              type="button"
              className="submit-button upload-button"
              onClick={handleOpenFilePicker}
              disabled={isUploading}
            >
              {isUploading ? "Uploading..." : "Upload PDF"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
