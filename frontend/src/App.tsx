import { useEffect, useState } from "react";
import "./App.css";

// One document item returned by the backend
type DocumentItem = {
  doc_id: string;
  original_filename: string;
};

// Full response shape from GET /documents
type DocumentsResponse = {
  items: DocumentItem[];
};

// Modes shown to the user in the UI
type UiQueryMode = "corpus" | "document" | "chat" | "auto";

// Modes actually supported by the backend
type BackendQueryMode = "grounded" | "chat" | "auto";

// Request body sent to POST /query
type QueryRequestPayload = {
  query: string;
  mode: BackendQueryMode;
  doc_ids?: string[];
};

// One retrieved source returned by POST /query
type QuerySource = {
  rank: number;
  chunk_id: string;
  doc_id?: string | null;
  source_file: string | null;
  original_filename?: string | null;
  page_start: number | null;
  page_end: number | null;
  rerank_score: number | null;
  fusion_score: number | null;
  headings: string[] | null;
  block_type: string | null;
};

// Response fields we currently use from POST /query
type QueryResponse = {
  answer: string;
  sources: QuerySource[];
};

type QueryStreamStart = {
  query: string;
  mode_used: string;
  fallback_reason: string | null;
  sources: QuerySource[];
  used_context_tokens: number;
  retrieved_chunk_count: number;
  retrieval_seconds: number;
};

type QueryStreamDone = {
  answer: string;
  timings: {
    retrieval_seconds: number;
    generation_seconds: number;
    total_seconds: number;
  };
};

type QueryStreamEvent =
  | { type: "start"; data: QueryStreamStart }
  | { type: "token"; data: { text: string } }
  | { type: "done"; data: QueryStreamDone }
  | { type: "error"; data: { message: string } };

export default function App() {
  // Stores all documents loaded from the backend
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  // Stores an error message if loading fails
  const [error, setError] = useState<string | null>(null);

  // Stores the currently selected document id
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  // Stores the single user-facing query mode
  const [uiMode, setUiMode] = useState<UiQueryMode>("corpus");

  // Stores the text typed by the user
  const [queryText, setQueryText] = useState("");

   // Stores the final answer returned by the backend
  const [answer, setAnswer] = useState("");

  // Stores retrieved sources for the right pane
  const [sources, setSources] = useState<QuerySource[]>([]);

  // Stores query request error
  const [queryError, setQueryError] = useState<string | null>(null);

  // Tracks whether a query request is running
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Keep evidence ordered by backend rank
  const sortedSources = [...sources].sort((left, right) => left.rank - right.rank);

  useEffect(() => {
    // Loads documents once when the component first renders
    async function loadDocuments() {
      try {
        // Call the FastAPI backend
        const response = await fetch("http://localhost:8000/documents");

        // Stop if backend returned an error status
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }

        // Convert JSON response into a typed object
        const data: DocumentsResponse = await response.json();

        // Save documents into component state
        setDocuments(data.items);
      } catch (err) {
        // Save a readable error message
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Failed to load documents");
        }
      }
    }

    loadDocuments();
  }, []);

  // Sends the current query to the backend and streams the answer incrementally
  async function handleSubmit() {
      // Basic guard: do nothing for empty input
      if (!queryText.trim()) {
        return;
      }

      // Single-document mode requires one selected document
      if (uiMode === "document" && !selectedDocId) {
        setQueryError("Select a document first.");
        return;
      }

      // Clear old request state before sending a new query
      setQueryError(null);
      setAnswer("");
      setSources([]);
      setIsSubmitting(true);

      try {
        // Default backend mode for corpus and single-document queries
        let backendMode: BackendQueryMode = "grounded";

        if (uiMode === "chat") {
          backendMode = "chat";
        } else if (uiMode === "auto") {
          backendMode = "auto";
        }

        // Add doc_ids only for modes that should use the selected document
        let docIds: string[] | undefined = undefined;

        if ((uiMode === "document" || uiMode === "auto") && selectedDocId) {
          docIds = [selectedDocId];
        }

        // Build request body for FastAPI
        const payload: QueryRequestPayload = {
          query: queryText,
          mode: backendMode,
          ...(docIds ? { doc_ids: docIds } : {}),
        };

        // Send POST /query/stream to the backend
        const response = await fetch("http://localhost:8000/query/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        // Stop on backend error response
        if (!response.ok) {
          throw new Error(`Query failed: ${response.status}`);
        }

        // Streaming requires a readable response body
        if (!response.body) {
          throw new Error("Streaming response body is missing.");
        }

        // Read the response as a byte stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        // Buffer keeps incomplete lines between chunks
        let buffer = "";

        while (true) {
          // Read the next chunk from the stream
          const { value, done } = await reader.read();

          // Exit once the backend closes the stream
          if (done) {
            break;
          }

          // Decode bytes into text and append to the buffer
          buffer += decoder.decode(value, { stream: true });

          // Split complete NDJSON lines and keep the unfinished tail
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();

            // Ignore empty lines
            if (!trimmed) {
              continue;
            }

            // Parse one streamed event from the backend
            const event: QueryStreamEvent = JSON.parse(trimmed);

            // Initial metadata event with sources and retrieval info
            if (event.type === "start") {
              setSources(event.data.sources);
              continue;
            }

            // Append the next streamed token chunk to the answer
            if (event.type === "token") {
              setAnswer((current) => current + event.data.text);
              continue;
            }

            // Surface backend-side streaming errors
            if (event.type === "error") {
              setQueryError(event.data.message);
              continue;
            }

            // Final event with the completed answer and timing info
            if (event.type === "done") {
              setAnswer(event.data.answer);
            }
          }
        }

        // Process any last buffered line after the stream ends
        const trailing = buffer.trim();

        if (trailing) {
          const event: QueryStreamEvent = JSON.parse(trailing);

          if (event.type === "start") {
            setSources(event.data.sources);
          } else if (event.type === "token") {
            setAnswer((current) => current + event.data.text);
          } else if (event.type === "error") {
            setQueryError(event.data.message);
          } else if (event.type === "done") {
            setAnswer(event.data.answer);
          }
        }
      } catch (err) {
        // Save a readable error message
        if (err instanceof Error) {
          setQueryError(err.message);
        } else {
          setQueryError("Failed to run query");
        }
      } finally {
        // Always stop loading state
        setIsSubmitting(false);
      }
    }

return (
  <div className="app-shell">
    <aside className="left-pane">
      <h2>Documents</h2>

      {/* Show error only if one exists */}
      {error && <p>{error}</p>}

      <ul className="document-list">
        {documents.map((document) => (
          <li key={document.doc_id}>
            <button
              type="button"
              className={
                document.doc_id === selectedDocId
                  ? "document-button selected"
                  : "document-button"
              }
              onClick={() => setSelectedDocId(document.doc_id)}
            >
              {document.original_filename}
            </button>
          </li>
        ))}
      </ul>
    </aside>

    <main className="center-pane">
      <h2>Query Workspace</h2>

      {/* Single user-facing mode selector */}
      <div className="scope-toggle">
        <button
          type="button"
          className={uiMode === "corpus" ? "scope-button selected" : "scope-button"}
          onClick={() => setUiMode("corpus")}
        >
          Corpus
        </button>

        <button
          type="button"
          className={uiMode === "document" ? "scope-button selected" : "scope-button"}
          onClick={() => setUiMode("document")}
        >
          Single Document
        </button>

        <button
          type="button"
          className={uiMode === "chat" ? "scope-button selected" : "scope-button"}
          onClick={() => setUiMode("chat")}
        >
          Chat
        </button>

        <button
          type="button"
          className={uiMode === "auto" ? "scope-button selected" : "scope-button"}
          onClick={() => setUiMode("auto")}
        >
          Auto
        </button>
      </div>

      {/* Controlled input: value comes from state, typing updates state */}
      <input
        type="text"
        className="query-input"
        placeholder="Ask a question about your documents"
        value={queryText}
        onChange={(event) => setQueryText(event.target.value)}
      />

      {/* Sends the current query to the backend */}
      <button type="button" className="submit-button" onClick={handleSubmit}>
        {isSubmitting ? "Running..." : "Ask"}
      </button>

      {/* Show query error only if one exists */}
      {queryError && <p>{queryError}</p>}

      {/* Show backend answer */}
      <div className="answer-box">
        <h3>Answer</h3>
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
          {answer || "No answer yet"}
        </pre>
      </div>
    </main>

    <aside className="right-pane">
      <h2>Evidence</h2>

      {/* Show a placeholder when no evidence is available */}
      {sortedSources.length === 0 && <p>No evidence yet</p>}

      <ul className="source-list">
        {sortedSources.map((source) => (
          <li key={source.chunk_id} className="source-item">
            <div>
              {/* Prefer the real uploaded filename from the backend */}
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
  </div>
);
}