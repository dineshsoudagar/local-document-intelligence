import { useEffect, useRef, useState } from "react";
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

// Response shape returned by POST /documents/upload
type DocumentUploadResponse = {
  message: string;
  deduplicated: boolean;
  document: DocumentItem;
};

type DocumentDeleteResponse = {
  message: string;
  doc_id: string;
};

// Modes shown to the user in the UI
type UiQueryMode = "corpus" | "document" | "chat" | "auto";

// Modes actually supported by the backend
type BackendQueryMode = "grounded" | "chat" | "auto";

// Request body sent to the query endpoint
type QueryRequestPayload = {
  query: string;
  mode: BackendQueryMode;
  doc_ids?: string[];
};

// One retrieved source returned by the query backend
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

// Initial metadata event sent before streamed answer text
type QueryStreamStart = {
  query: string;
  mode_used: string;
  fallback_reason: string | null;
  sources: QuerySource[];
  used_context_tokens: number;
  retrieved_chunk_count: number;
  retrieval_seconds: number;
};

// Final event sent after streaming completes
type QueryStreamDone = {
  answer: string;
  timings: {
    retrieval_seconds: number;
    generation_seconds: number;
    total_seconds: number;
  };
};

// All supported event shapes in the NDJSON stream
type QueryStreamEvent =
  | { type: "start"; data: QueryStreamStart }
  | { type: "token"; data: { text: string } }
  | { type: "done"; data: QueryStreamDone }
  | { type: "error"; data: { message: string } };

// Frontend request lifecycle states
type QueryStatus = "idle" | "retrieving" | "generating" | "error";

export default function App() {
  // Stores all documents loaded from the backend
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  // Stores an error message if document loading fails
  const [error, setError] = useState<string | null>(null);

  // Stores the currently selected document id
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  // Stores the single user-facing query mode
  const [uiMode, setUiMode] = useState<UiQueryMode>("corpus");

  // Stores the text typed by the user
  const [queryText, setQueryText] = useState("");

  // Stores the streamed answer text
  const [answer, setAnswer] = useState("");

  // Stores retrieved sources for the right pane
  const [sources, setSources] = useState<QuerySource[]>([]);

  // Stores query request error
  const [queryError, setQueryError] = useState<string | null>(null);

  // Tracks whether a query request is running
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Tracks whether a document upload is running
  const [isUploading, setIsUploading] = useState(false);

  // Stores upload-specific errors
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Tracks which document is currently being deleted
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);

  // Tracks drag-over styling for the drop zone
  const [isDragOver, setIsDragOver] = useState(false);

  // Hidden file input used by the upload button
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Tracks the current request lifecycle state
  const [queryStatus, setQueryStatus] = useState<QueryStatus>("idle");

  // Stores the backend-resolved mode for the current answer
  const [resolvedMode, setResolvedMode] = useState<string | null>(null);

  // Stores the auto-mode fallback reason when provided by the backend
  const [fallbackReason, setFallbackReason] = useState<string | null>(null);

  // Keeps the active request controller so streaming can be cancelled
  const abortControllerRef = useRef<AbortController | null>(null);

  // Keep evidence ordered by backend rank
  const sortedSources = [...sources].sort((left, right) => left.rank - right.rank);

  function isFileDrag(event: DragEvent | React.DragEvent<HTMLDivElement>) {
    return Array.from(event.dataTransfer?.types ?? []).includes("Files");
  }

  function getDraggedFile(event: React.DragEvent<HTMLDivElement>) {
    const item = Array.from(event.dataTransfer.items ?? []).find(
      (candidate) => candidate.kind === "file",
    );

    if (item) {
      return item.getAsFile();
    }

    return event.dataTransfer.files?.[0] ?? null;
  }

  // Loads the current document list from the backend
  async function loadDocuments() {
    try {
      setError(null);

      const response = await fetch("http://localhost:8000/documents");

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data: DocumentsResponse = await response.json();
      setDocuments(data.items);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to load documents");
      }
    }
  }

  useEffect(() => {
    loadDocuments();

    // Cancel any in-flight request if the component unmounts
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
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

  // Stops the current in-flight streamed request
  function handleStop() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsSubmitting(false);
    setQueryStatus("idle");
  }

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

    // Cancel any previous in-flight request before starting a new one
    abortControllerRef.current?.abort();

    // Create a fresh controller for this request
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Clear old request state before sending a new query
    setQueryError(null);
    setAnswer("");
    setSources([]);
    setResolvedMode(null);
    setFallbackReason(null);
    setIsSubmitting(true);
    setQueryStatus("retrieving");

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
        signal: controller.signal,
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
            setResolvedMode(event.data.mode_used);
            setFallbackReason(event.data.fallback_reason);
            setQueryStatus("generating");
            continue;
          }

          // Append the next streamed token text to the answer
          if (event.type === "token") {
            setAnswer((current) => current + event.data.text);
            continue;
          }

          // Stop on backend-side streaming errors
          if (event.type === "error") {
            throw new Error(event.data.message);
          }

          // Final event carries the completed answer and timing info
          if (event.type === "done") {
            setAnswer(event.data.answer);
            setQueryStatus("idle");
          }
        }
      }

      // Process any last buffered line after the stream ends
      const trailing = buffer.trim();

      if (trailing) {
        const event: QueryStreamEvent = JSON.parse(trailing);

        if (event.type === "start") {
          setSources(event.data.sources);
          setResolvedMode(event.data.mode_used);
          setFallbackReason(event.data.fallback_reason);
          setQueryStatus("generating");
        } else if (event.type === "token") {
          setAnswer((current) => current + event.data.text);
        } else if (event.type === "error") {
          throw new Error(event.data.message);
        } else if (event.type === "done") {
          setAnswer(event.data.answer);
          setQueryStatus("idle");
        }
      }
    } catch (err) {
      // Treat user cancellation as a normal stop, not an error
      if (err instanceof DOMException && err.name === "AbortError") {
        setQueryStatus("idle");
        return;
      }

      // Save a readable error message
      if (err instanceof Error) {
        setQueryError(err.message);
      } else {
        setQueryError("Failed to run query");
      }

      setQueryStatus("error");
    } finally {
      // Clear controller and always stop loading state
      abortControllerRef.current = null;
      setIsSubmitting(false);
    }
  }

  // Uploads one PDF file to the backend and refreshes the document list
  async function handleUploadFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported right now.");
      return;
    }

    setUploadError(null);
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const data: DocumentUploadResponse = await response.json();

      // Refresh the left pane so the new document appears immediately
      await loadDocuments();

      // Auto-select the uploaded document for convenience
      setSelectedDocId(data.document.doc_id);
    } catch (err) {
      if (err instanceof Error) {
        setUploadError(err.message);
      } else {
        setUploadError("Failed to upload document");
      }
    } finally {
      setIsUploading(false);
    }
  }

  async function handleDeleteDocument(document: DocumentItem) {
    const confirmed = window.confirm(
      `Delete "${document.original_filename}" from the uploaded documents?`,
    );

    if (!confirmed) {
      return;
    }

    setUploadError(null);
    setDeletingDocId(document.doc_id);

    try {
      const response = await fetch(
        `http://localhost:8000/documents/${document.doc_id}`,
        {
          method: "DELETE",
        },
      );

      if (!response.ok) {
        throw new Error(`Delete failed: ${response.status}`);
      }

      const data: DocumentDeleteResponse = await response.json();

      setDocuments((current) =>
        current.filter((item) => item.doc_id !== data.doc_id),
      );

      if (selectedDocId === data.doc_id) {
        setSelectedDocId(null);
      }
    } catch (err) {
      if (err instanceof Error) {
        setUploadError(err.message);
      } else {
        setUploadError("Failed to delete document");
      }
    } finally {
      setDeletingDocId(null);
    }
  }

  // Handles file selection from the hidden file input
  async function handleFileInputChange(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    await handleUploadFile(file);

    // Reset input so the same file can be selected again later if needed
    event.target.value = "";
  }

  // Opens the hidden native file picker
  function handleOpenFilePicker() {
    fileInputRef.current?.click();
  }

  // Prevent browser default so dropping a file stays inside the app
  function handleDragOver(event: React.DragEvent<HTMLDivElement>) {
    if (!isFileDrag(event)) {
      return;
    }

    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsDragOver(true);
  }

  // Remove drag highlight when the file leaves the drop zone
  function handleDragLeave(event: React.DragEvent<HTMLDivElement>) {
    if (!isFileDrag(event)) {
      return;
    }

    event.preventDefault();

    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }

    setIsDragOver(false);
  }

  // Accept one dropped file and send it through the same upload path
  async function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    if (!isFileDrag(event)) {
      return;
    }

    event.preventDefault();
    setIsDragOver(false);

    const file = getDraggedFile(event);
    if (!file) {
      setUploadError("No file was detected in the drop operation.");
      return;
    }

    await handleUploadFile(file);
  }

  return (
    <div className="app-shell">
      <aside
        className={isDragOver ? "left-pane drag-over" : "left-pane"}
        onDragEnter={handleDragOver}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <h2>Documents</h2>
        {/* Hidden input used by the Upload button */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          style={{ display: "none" }}
          onChange={handleFileInputChange}
        />

        {/* Upload actions for button click and drag/drop */}
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

        {/* Show upload error only if one exists */}
        {uploadError && <p>{uploadError}</p>}
        {/* Show error only if one exists */}
        {error && <p>{error}</p>}

        <ul className="document-list">
          {documents.map((document) => (
            <li key={document.doc_id}>
              <div className="document-row">
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

                <button
                  type="button"
                  className="delete-document-button"
                  onClick={() => handleDeleteDocument(document)}
                  disabled={deletingDocId === document.doc_id}
                  aria-label={`Delete ${document.original_filename}`}
                >
                  {deletingDocId === document.doc_id ? "..." : "Delete"}
                </button>
              </div>
            </li>
          ))}
        </ul>
      </aside>

      <main className="center-pane">
        <div className="center-header">
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

          {/* Show current request status and backend routing details */}
          {(isSubmitting || resolvedMode || fallbackReason || queryStatus !== "idle") && (
            <div className="query-meta">
              <div>Status: {queryStatus}</div>
              {resolvedMode && <div>Mode used: {resolvedMode}</div>}
              {fallbackReason && <div>Fallback: {fallbackReason}</div>}
            </div>
          )}

          {/* Show query error only if one exists */}
          {queryError && <p className="query-error">{queryError}</p>}
        </div>

        <div className="answer-box">
          <h3>Answer</h3>
          <pre className="answer-text">
            {answer || "No answer yet"}
          </pre>
        </div>

        <div className="composer-section">
          <label className="composer-label" htmlFor="query-input">
            Query
          </label>

          {/* Query input and actions stay on one row */}
          <div className="composer-row">
            <input
              id="query-input"
              type="text"
              className="query-input"
              placeholder="Ask a question about your documents"
              value={queryText}
              onChange={(event) => setQueryText(event.target.value)}
              onKeyDown={(event) => {
                // Submit on Enter when no request is running
                if (event.key === "Enter" && !isSubmitting) {
                  handleSubmit();
                }
              }}
            />

            <button
              type="button"
              className="submit-button ask-button"
              onClick={handleSubmit}
              disabled={isSubmitting}
            >
              {isSubmitting ? "Running..." : "Ask"}
            </button>

            <button
              type="button"
              className="submit-button stop-button"
              onClick={handleStop}
              disabled={!isSubmitting}
            >
              Stop
            </button>
          </div>
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
