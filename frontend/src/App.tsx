import { useEffect, useRef, useState } from "react";
import "./App.css";
import { deleteDocument, fetchDocuments, streamQuery, uploadDocument } from "./api";
import { DocumentsPane } from "./components/DocumentsPane";
import { EvidencePane } from "./components/EvidencePane";
import { QueryPane } from "./components/QueryPane";
import type {
  BackendQueryMode,
  DocumentItem,
  QueryRequestPayload,
  QuerySource,
  QueryStatus,
  UiQueryMode,
} from "./types";

export default function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [uiMode, setUiMode] = useState<UiQueryMode>("corpus");
  const [queryText, setQueryText] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<QuerySource[]>([]);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [openMenuDocId, setOpenMenuDocId] = useState<string | null>(null);
  const [queryStatus, setQueryStatus] = useState<QueryStatus>("idle");
  const [resolvedMode, setResolvedMode] = useState<string | null>(null);
  const [fallbackReason, setFallbackReason] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

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
    loadDocuments();

    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    function handleWindowClick() {
      setOpenMenuDocId(null);
    }

    window.addEventListener("click", handleWindowClick);

    return () => {
      window.removeEventListener("click", handleWindowClick);
    };
  }, []);

  function handleStop() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsSubmitting(false);
    setQueryStatus("idle");
  }

  async function handleSubmit() {
    if (!queryText.trim()) {
      return;
    }

    if (uiMode === "document" && !selectedDocId) {
      setQueryError("Select a document first.");
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setQueryError(null);
    setAnswer("");
    setSources([]);
    setResolvedMode(null);
    setFallbackReason(null);
    setIsSubmitting(true);
    setQueryStatus("retrieving");

    try {
      let backendMode: BackendQueryMode = "grounded";

      if (uiMode === "chat") {
        backendMode = "chat";
      } else if (uiMode === "auto") {
        backendMode = "auto";
      }

      let docIds: string[] | undefined;

      if ((uiMode === "document" || uiMode === "auto") && selectedDocId) {
        docIds = [selectedDocId];
      }

      const payload: QueryRequestPayload = {
        query: queryText,
        mode: backendMode,
        ...(docIds ? { doc_ids: docIds } : {}),
      };

      await streamQuery(payload, controller.signal, {
        onStart: (data) => {
          setSources(data.sources);
          setResolvedMode(data.mode_used);
          setFallbackReason(data.fallback_reason);
          setQueryStatus("generating");
        },
        onToken: (text) => {
          setAnswer((current) => current + text);
        },
        onDone: (data) => {
          setAnswer(data.answer);
          setQueryStatus("idle");
        },
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setQueryStatus("idle");
        return;
      }

      if (err instanceof Error) {
        setQueryError(err.message);
      } else {
        setQueryError("Failed to run query");
      }

      setQueryStatus("error");
    } finally {
      abortControllerRef.current = null;
      setIsSubmitting(false);
    }
  }

  async function handleUploadFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported right now.");
      return;
    }

    setUploadError(null);
    setIsUploading(true);

    try {
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
      setDocuments((current) => current.filter((item) => item.doc_id !== data.doc_id));

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
      <DocumentsPane
        documents={documents}
        selectedDocId={selectedDocId}
        isUploading={isUploading}
        uploadError={uploadError}
        error={error}
        deletingDocId={deletingDocId}
        openMenuDocId={openMenuDocId}
        onSelectDocument={setSelectedDocId}
        onDeleteDocument={handleDeleteDocument}
        onToggleMenu={setOpenMenuDocId}
        onUploadFile={handleUploadFile}
      />

      <QueryPane
        uiMode={uiMode}
        queryStatus={queryStatus}
        resolvedMode={resolvedMode}
        fallbackReason={fallbackReason}
        queryError={queryError}
        answer={answer}
        queryText={queryText}
        isSubmitting={isSubmitting}
        onModeChange={setUiMode}
        onQueryTextChange={setQueryText}
        onSubmit={handleSubmit}
        onStop={handleStop}
      />

      <EvidencePane sources={sources} />
    </div>
  );
}
