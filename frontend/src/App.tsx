import { useEffect, useRef, useState } from "react";
import "./App.css";
import { deleteDocument, fetchDocuments, streamQuery, uploadDocument } from "./api";
import { DocumentsPane } from "./components/DocumentsPane";
import { QueryPane } from "./components/QueryPane";
import type {
  ChatMessage,
  BackendQueryMode,
  DocumentItem,
  QueryRequestPayload,
  QueryStatus,
  UiQueryMode,
} from "./types";

export default function App() {
  // App owns the shared frontend state and passes focused slices into each pane.
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const storedTheme = window.localStorage.getItem("theme");
    if (storedTheme === "light" || storedTheme === "dark") {
      return storedTheme;
    }

    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [uiMode, setUiMode] = useState<UiQueryMode>("corpus");
  const [queryText, setQueryText] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [openMenuDocId, setOpenMenuDocId] = useState<string | null>(null);
  const [queryStatus, setQueryStatus] = useState<QueryStatus>("idle");
  const abortControllerRef = useRef<AbortController | null>(null);

  function updateMessage(
    messageId: string,
    updater: (message: ChatMessage) => ChatMessage,
  ) {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId ? updater(message) : message,
      ),
    );
  }

  function createMessageId(prefix: "user" | "assistant") {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  // Pull the latest uploaded documents from the backend so the left pane stays in sync.
  async function loadDocuments() {
    try {
      setError(null);
      const items = await fetchDocuments();
      setDocuments(items);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to load documents");
      }
    }
  }

  useEffect(() => {
    // Load initial document data once when the app mounts.
    loadDocuments();

    return () => {
      // Cancel any in-flight query if the page is being torn down.
      abortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    // Close any open document action menu when the user clicks elsewhere.
    function handleWindowClick() {
      setOpenMenuDocId(null);
    }

    window.addEventListener("click", handleWindowClick);

    return () => {
      window.removeEventListener("click", handleWindowClick);
    };
  }, []);

  useEffect(() => {
    document.body.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    // Keep selection aligned with the current document list.
    if (documents.length === 0) {
      if (selectedDocId !== null) {
        setSelectedDocId(null);
      }
      return;
    }

    const hasSelectedDocument = selectedDocId
      ? documents.some((document) => document.doc_id === selectedDocId)
      : false;

    if (hasSelectedDocument) {
      return;
    }

    if (documents.length === 1) {
      setSelectedDocId(documents[0].doc_id);
      return;
    }

    if (selectedDocId !== null) {
      setSelectedDocId(null);
    }
  }, [documents, selectedDocId]);

  // Stop the active streaming query and reset the UI back to idle.
  function handleStop() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsSubmitting(false);
    setQueryStatus("idle");
  }

  async function handleSubmit() {
    // Ignore empty submissions so the backend is only called with real input.
    const trimmedQuery = queryText.trim();
    const effectiveSelectedDocId =
      selectedDocId ?? (documents.length === 1 ? documents[0].doc_id : null);

    if (!trimmedQuery) {
      return;
    }

    // Document-only mode requires an explicit document selection first.
    if (uiMode === "document" && !effectiveSelectedDocId) {
      setQueryError("Select a document first.");
      return;
    }

    // Replace any older request so only the newest query continues streaming.
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const userMessageId = createMessageId("user");
    const assistantMessageId = createMessageId("assistant");

    setQueryError(null);
    setIsSubmitting(true);
    setQueryStatus("retrieving");
    setQueryText("");
    setMessages((current) => [
      ...current,
      {
        id: userMessageId,
        role: "user",
        content: trimmedQuery,
        status: "complete",
      },
      {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        status: "streaming",
        sources: [],
      },
    ]);

    try {
      // Translate the UI-specific mode labels into the backend API contract.
      let backendMode: BackendQueryMode = "grounded";

      if (uiMode === "chat") {
        backendMode = "chat";
      } else if (uiMode === "auto") {
        backendMode = "auto";
      }

      let docIds: string[] | undefined;

      // When a document is selected, scope retrieval to that document if the mode allows it.
      if ((uiMode === "document" || uiMode === "auto") && effectiveSelectedDocId) {
        docIds = [effectiveSelectedDocId];
      }

      // Build the request payload and include doc_ids only when a document filter exists.
      const payload: QueryRequestPayload = {
        query: trimmedQuery,
        mode: backendMode,
        ...(docIds ? { doc_ids: docIds } : {}),
      };

      await streamQuery(payload, controller.signal, {
        onStart: (data) => {
          // The start event tells us which evidence was chosen before tokens arrive.
          updateMessage(assistantMessageId, (message) => ({
            ...message,
            sources: data.sources,
          }));
          setQueryStatus("generating");
        },
        onToken: (text) => {
          // Append streamed tokens so the answer appears progressively.
          updateMessage(assistantMessageId, (message) => ({
            ...message,
            content: message.content + text,
            status: "streaming",
          }));
        },
        onDone: (data) => {
          // Replace the streamed draft with the backend's final answer snapshot.
          updateMessage(assistantMessageId, (message) => ({
            ...message,
            content: data.answer,
            status: "complete",
          }));
          setQueryStatus("idle");
        },
      });
    } catch (err) {
      // User-triggered cancellation is an expected path, not an error state.
      if (err instanceof DOMException && err.name === "AbortError") {
        updateMessage(assistantMessageId, (message) => ({
          ...message,
          content: message.content || "Response stopped.",
          status: "complete",
        }));
        setQueryStatus("idle");
        return;
      }

      // Surface backend or network failures inside the query pane.
      const message = err instanceof Error ? err.message : "Failed to run query";

      if (err instanceof Error) {
        setQueryError(err.message);
      } else {
        setQueryError("Failed to run query");
      }

      updateMessage(assistantMessageId, (chatMessage) => ({
        ...chatMessage,
        content:
          chatMessage.content ||
          `I couldn't complete that request.\n\n${message}`,
        status: "error",
      }));
      setQueryStatus("error");
    } finally {
      // Release the controller so the next query starts from a clean state.
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      setIsSubmitting(false);
    }
  }

  // Upload one PDF, then refresh the document list and select the new item.
  async function handleUploadFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported right now.");
      return;
    }

    setUploadError(null);
    setIsUploading(true);

    try {
      // Refresh after upload so any backend-generated metadata is reflected in the UI.
      const data = await uploadDocument(file);
      await loadDocuments();
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

  // Delete a document from the backend and remove it from local state after confirmation.
  async function handleDeleteDocument(document: DocumentItem) {
    setOpenMenuDocId(null);

    const confirmed = window.confirm(
      `Delete "${document.original_filename}" from the uploaded documents?`,
    );

    if (!confirmed) {
      return;
    }

    setUploadError(null);
    setDeletingDocId(document.doc_id);

    try {
      const data = await deleteDocument(document.doc_id);
      // Update the list locally so the UI responds immediately after deletion.
      setDocuments((current) => current.filter((item) => item.doc_id !== data.doc_id));

      // Clear the selection if the deleted document was the active one.
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

  return (
    <div className="app-shell">
      {/* Left pane handles upload, selection, and document actions. */}
      <DocumentsPane
        documents={documents}
        selectedDocId={selectedDocId}
        error={error}
        deletingDocId={deletingDocId}
        openMenuDocId={openMenuDocId}
        onSelectDocument={setSelectedDocId}
        onDeleteDocument={handleDeleteDocument}
        onToggleMenu={setOpenMenuDocId}
      />

      {/* Center pane owns query input, answer display, and query status messaging. */}
      <QueryPane
        documents={documents}
        messages={messages}
        theme={theme}
        uiMode={uiMode}
        queryStatus={queryStatus}
        queryError={queryError}
        queryText={queryText}
        isSubmitting={isSubmitting}
        isUploading={isUploading}
        uploadError={uploadError}
        onModeChange={setUiMode}
        onQueryTextChange={setQueryText}
        onSubmit={handleSubmit}
        onStop={handleStop}
        onToggleTheme={() =>
          setTheme((currentTheme) => (currentTheme === "light" ? "dark" : "light"))
        }
        onUploadFile={handleUploadFile}
      />
    </div>
  );
}
