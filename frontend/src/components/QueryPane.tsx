import type { QueryStatus, UiQueryMode } from "../types";

type QueryPaneProps = {
  uiMode: UiQueryMode;
  queryStatus: QueryStatus;
  resolvedMode: string | null;
  fallbackReason: string | null;
  queryError: string | null;
  answer: string;
  queryText: string;
  isSubmitting: boolean;
  onModeChange: (mode: UiQueryMode) => void;
  onQueryTextChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
};

export function QueryPane({
  uiMode,
  queryStatus,
  resolvedMode,
  fallbackReason,
  queryError,
  answer,
  queryText,
  isSubmitting,
  onModeChange,
  onQueryTextChange,
  onSubmit,
  onStop,
}: QueryPaneProps) {
  return (
    <main className="center-pane">
      <div className="center-header">
        <h2>Query Workspace</h2>

        <div className="scope-toggle">
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
            Single Document
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
            className={uiMode === "auto" ? "scope-button selected" : "scope-button"}
            onClick={() => onModeChange("auto")}
          >
            Auto
          </button>
        </div>

        {(isSubmitting || resolvedMode || fallbackReason || queryStatus !== "idle") && (
          <div className="query-meta">
            <div>Status: {queryStatus}</div>
            {resolvedMode && <div>Mode used: {resolvedMode}</div>}
            {fallbackReason && <div>Fallback: {fallbackReason}</div>}
          </div>
        )}

        {queryError && <p className="query-error">{queryError}</p>}
      </div>

      <div className="answer-box">
        <h3>Answer</h3>
        <pre className="answer-text">{answer || "No answer yet"}</pre>
      </div>

      <div className="composer-section">
        <label className="composer-label" htmlFor="query-input">
          Query
        </label>

        <div className="composer-row">
          <input
            id="query-input"
            type="text"
            className="query-input"
            placeholder="Ask a question about your documents"
            value={queryText}
            onChange={(event) => onQueryTextChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !isSubmitting) {
                onSubmit();
              }
            }}
          />

          <button
            type="button"
            className="submit-button ask-button"
            onClick={onSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? "Running..." : "Ask"}
          </button>

          <button
            type="button"
            className="submit-button stop-button"
            onClick={onStop}
            disabled={!isSubmitting}
          >
            Stop
          </button>
        </div>
      </div>
    </main>
  );
}
