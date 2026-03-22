import { useEffect, useRef } from "react";
import type { QueryStatus, UiQueryMode } from "../types";
import type { ChatMessage } from "../types";

type QueryPaneProps = {
  messages: ChatMessage[];
  uiMode: UiQueryMode;
  queryStatus: QueryStatus;
  resolvedMode: string | null;
  fallbackReason: string | null;
  queryError: string | null;
  queryText: string;
  isSubmitting: boolean;
  onModeChange: (mode: UiQueryMode) => void;
  onQueryTextChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
};

export function QueryPane({
  messages,
  uiMode,
  queryStatus,
  resolvedMode,
  fallbackReason,
  queryError,
  queryText,
  isSubmitting,
  onModeChange,
  onQueryTextChange,
  onSubmit,
  onStop,
}: QueryPaneProps) {
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, queryStatus]);

  return (
    <main className="center-pane">
      <div className="center-header">
        <div className="center-header-copy">
          <h2>Chat Workspace</h2>
          <p>Ask questions about your uploaded documents and review the responses below.</p>
        </div>

        {/* These buttons switch how the backend should interpret the query. */}
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

        {/* Show live query metadata once a request starts or when the backend resolves auto mode. */}
        {(isSubmitting || resolvedMode || fallbackReason || queryStatus !== "idle") && (
          <div className="query-meta">
            <div>Status: {queryStatus}</div>
            {resolvedMode && <div>Mode used: {resolvedMode}</div>}
            {fallbackReason && <div>Fallback: {fallbackReason}</div>}
          </div>
        )}

        {queryError && <p className="query-error">{queryError}</p>}
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
                    {message.content || (message.status === "streaming" ? "Thinking..." : "")}
                  </div>
                </div>
              </div>
            ))}
            <div ref={transcriptEndRef} />
          </div>
        )}
      </div>

      <div className="composer-section">
        <label className="composer-label" htmlFor="query-input">
          Message
        </label>

        <div className="composer-row">
          <textarea
            id="query-input"
            className="query-input"
            placeholder="Ask a question about your documents"
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

          {/* Ask starts a new query; Stop cancels the current stream. */}
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
