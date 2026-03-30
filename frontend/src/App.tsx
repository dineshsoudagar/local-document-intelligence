import { useEffect, useRef, useState } from "react";
import "./App.css";
import {
  cancelSetup,
  deleteDocument,
  fetchDocuments,
  fetchSetupOptions,
  fetchSetupStatus,
  retrySetup,
  startSetup,
  streamQuery,
  uploadDocument,
} from "./api";
import { DocumentsPane } from "./components/DocumentsPane";
import { QueryPane } from "./components/QueryPane";
import { SetupPane } from "./components/SetupPane";
import type {
  ChatMessage,
  BackendQueryMode,
  DocumentItem,
  QueryRequestPayload,
  QueryStatus,
  SetupOptions,
  SetupStatus,
  UiQueryMode,
} from "./types";

export default function App() {
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
  const [isFetchingDocuments, setIsFetchingDocuments] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [uiMode, setUiMode] = useState<UiQueryMode>("auto");
  const [queryText, setQueryText] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [openMenuDocId, setOpenMenuDocId] = useState<string | null>(null);
  const [queryStatus, setQueryStatus] = useState<QueryStatus>("idle");
  const [setupOptions, setSetupOptions] = useState<SetupOptions | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [isSetupLoading, setIsSetupLoading] = useState(true);
  const [isReconfiguringSetup, setIsReconfiguringSetup] = useState(false);
  const [isRuntimeSwitching, setIsRuntimeSwitching] = useState(false);
  const [setupHandoffRequested, setSetupHandoffRequested] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const hasTriggeredManagedHandoffRef = useRef(false);

  const isRuntimeReady = setupStatus?.install_state === "ready";
  const hasDesktopManagedHandoffApi =
    typeof window !== "undefined" &&
    typeof window.pywebview?.api?.switchToManagedRuntime === "function";
  const shouldAutoSwitchToManagedRuntime =
    hasDesktopManagedHandoffApi &&
    setupHandoffRequested &&
    setupStatus?.install_state === "ready";

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

  function wait(ms: number) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  async function refreshSetupState() {
    const [statusPayload, optionsPayload] = await Promise.all([
      fetchSetupStatus(),
      fetchSetupOptions(),
    ]);
    setSetupStatus(statusPayload);
    setSetupOptions(optionsPayload);
    return statusPayload;
  }

  async function fetchAndStoreDocuments() {
    const items = await fetchDocuments();
    setDocuments(items);
    return items;
  }

  async function fetchDocumentsWithRetry() {
    let lastError: unknown = null;

    for (let attempt = 0; attempt < 10; attempt += 1) {
      try {
        return await fetchAndStoreDocuments();
      } catch (err) {
        lastError = err;
        if (attempt < 9) {
          await wait(1000);
        }
      }
    }

    throw lastError instanceof Error ? lastError : new Error("Failed to fetch documents");
  }

  async function loadDocuments() {
    setIsFetchingDocuments(true);
    try {
      setError(null);
      await fetchDocumentsWithRetry();
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to fetch documents");
      }
    } finally {
      setIsFetchingDocuments(false);
    }
  }

  async function pollDocumentsUntilVisible(docId: string) {
    setIsFetchingDocuments(true);

    try {
      setError(null);

      for (let attempt = 0; attempt < 20; attempt += 1) {
        const items = await fetchDocumentsWithRetry();
        if (items.some((document) => document.doc_id === docId)) {
          return true;
        }

        await wait(1000);
      }
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to fetch documents");
      }
    } finally {
      setIsFetchingDocuments(false);
    }

    return false;
  }

  useEffect(() => {
    let isMounted = true;

    async function loadInitialState() {
      setIsSetupLoading(true);
      try {
        setSetupError(null);
        const statusPayload = await refreshSetupState();
        if (!isMounted) {
          return;
        }

        if (statusPayload.install_state === "ready") {
          await loadDocuments();
        }
      } catch (err) {
        if (!isMounted) {
          return;
        }

        setSetupError(err instanceof Error ? err.message : "Failed to load setup state.");
      } finally {
        if (isMounted) {
          setIsSetupLoading(false);
        }
      }
    }

    void loadInitialState();

    return () => {
      isMounted = false;
      abortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!setupStatus?.is_busy) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshSetupState().catch((err) => {
        setSetupError(err instanceof Error ? err.message : "Failed to refresh setup status.");
      });
    }, 2000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [setupStatus?.is_busy]);

  useEffect(() => {
    if (
      setupStatus?.install_state !== "ready" ||
      isRuntimeSwitching ||
      shouldAutoSwitchToManagedRuntime
    ) {
      return;
    }

    if (documents.length === 0) {
      void loadDocuments();
    }
  }, [setupStatus?.install_state, isRuntimeSwitching, shouldAutoSwitchToManagedRuntime]);

  useEffect(() => {
    if (!shouldAutoSwitchToManagedRuntime || hasTriggeredManagedHandoffRef.current) {
      return;
    }

    hasTriggeredManagedHandoffRef.current = true;
    setIsRuntimeSwitching(true);
    setSetupError(null);
    setQueryError(null);
    setUploadError(null);

    const desktopApi = window.pywebview?.api;
    if (!desktopApi) {
      hasTriggeredManagedHandoffRef.current = false;
      setIsRuntimeSwitching(false);
      return;
    }

    void desktopApi
      .switchToManagedRuntime()
      .then(() => {
        setIsRuntimeSwitching(false);
        setSetupHandoffRequested(false);
      })
      .catch((err) => {
        hasTriggeredManagedHandoffRef.current = false;
        setIsRuntimeSwitching(false);
        setSetupError(
          err instanceof Error
            ? err.message
            : "Failed to switch to the managed runtime.",
        );
      });
  }, [shouldAutoSwitchToManagedRuntime]);

  useEffect(() => {
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
    setSelectedDocIds((current) => {
      if (current.length === 0) {
        return current;
      }

      const availableDocIds = new Set(documents.map((document) => document.doc_id));
      const nextSelectedDocIds = current.filter((docId) => availableDocIds.has(docId));

      return nextSelectedDocIds.length === current.length ? current : nextSelectedDocIds;
    });
  }, [documents]);

  function handleStop() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsSubmitting(false);
    setQueryStatus("idle");
  }

  function handleOpenSetupSettings() {
    handleStop();
    setQueryError(null);
    setUploadError(null);
    setSetupError(null);
    setIsReconfiguringSetup(true);
  }

  function handleReturnToWorkspace() {
    setIsReconfiguringSetup(false);
  }

  async function handleStartSetup(payload: {
    generator_key: string;
    embedding_key: string;
    generator_load_preset: string;
    torch_variant: string;
  }) {
    setSetupError(null);
    setSetupHandoffRequested(false);
    hasTriggeredManagedHandoffRef.current = false;
    const nextStatus = await startSetup(payload);
    setSetupStatus(nextStatus);
    if (nextStatus.install_state === "installing" || nextStatus.install_state === "ready") {
      setSetupHandoffRequested(true);
    }
  }

  async function handleRetrySetup() {
    setSetupError(null);
    setSetupHandoffRequested(false);
    hasTriggeredManagedHandoffRef.current = false;
    const nextStatus = await retrySetup();
    setSetupStatus(nextStatus);
    if (nextStatus.install_state === "installing" || nextStatus.install_state === "ready") {
      setSetupHandoffRequested(true);
    }
  }

  async function handleCancelSetup() {
    setSetupError(null);
    setSetupHandoffRequested(false);
    hasTriggeredManagedHandoffRef.current = false;
    const nextStatus = await cancelSetup();
    setSetupStatus(nextStatus);
  }

  useEffect(() => {
    if (!setupHandoffRequested || !setupStatus || setupStatus.is_busy) {
      return;
    }

    if (setupStatus.install_state !== "ready") {
      setSetupHandoffRequested(false);
      hasTriggeredManagedHandoffRef.current = false;
    }
  }, [setupHandoffRequested, setupStatus]);

  async function handleSubmit() {
    if (!isRuntimeReady) {
      setQueryError("Complete setup before sending queries.");
      return;
    }

    const trimmedQuery = queryText.trim();
    const effectiveSelectedDocIds = selectedDocIds;

    if (!trimmedQuery) {
      return;
    }

    if (uiMode === "document" && effectiveSelectedDocIds.length === 0) {
      setQueryError("Select at least one document first.");
      return;
    }

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
      let backendMode: BackendQueryMode = "grounded";

      if (uiMode === "chat") {
        backendMode = "chat";
      } else if (uiMode === "auto") {
        backendMode = "auto";
      }

      let docIds: string[] | undefined;

      if ((uiMode === "document" || uiMode === "auto") && effectiveSelectedDocIds.length > 0) {
        docIds = effectiveSelectedDocIds;
      }

      const payload: QueryRequestPayload = {
        query: trimmedQuery,
        mode: backendMode,
        ...(docIds ? { doc_ids: docIds } : {}),
      };

      await streamQuery(payload, controller.signal, {
        onStart: (data) => {
          updateMessage(assistantMessageId, (message) => ({
            ...message,
            sources: data.sources,
          }));
          setQueryStatus("generating");
        },
        onToken: (text) => {
          updateMessage(assistantMessageId, (message) => ({
            ...message,
            content: message.content + text,
            status: "streaming",
          }));
        },
        onDone: (data) => {
          updateMessage(assistantMessageId, (message) => ({
            ...message,
            content: data.answer,
            status: "complete",
          }));
          setQueryStatus("idle");
        },
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        updateMessage(assistantMessageId, (message) => ({
          ...message,
          content: message.content || "Response stopped.",
          status: "complete",
        }));
        setQueryStatus("idle");
        return;
      }

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
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      setIsSubmitting(false);
    }
  }

  async function handleUploadFile(file: File) {
    if (!isRuntimeReady) {
      setUploadError("Complete setup before uploading documents.");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported right now.");
      return;
    }

    setUploadError(null);
    setIsUploading(true);

    try {
      const data = await uploadDocument(file);
      const foundDocument = await pollDocumentsUntilVisible(data.document.doc_id);
      if (!foundDocument) {
        setDocuments((current) => {
          if (current.some((document) => document.doc_id === data.document.doc_id)) {
            return current;
          }

          return [...current, data.document];
        });
      }
      setSelectedDocIds([data.document.doc_id]);
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
      setSelectedDocIds((current) => current.filter((docId) => docId !== data.doc_id));
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

  if (isSetupLoading) {
    return <div className="setup-loading">Loading desktop runtime setup...</div>;
  }

  if (isRuntimeSwitching || shouldAutoSwitchToManagedRuntime) {
    return (
      <div className="setup-loading">
        Switching to the managed runtime...
      </div>
    );
  }

  if (!isRuntimeReady || isReconfiguringSetup) {
    return (
      <SetupPane
        options={setupOptions}
        status={setupStatus}
        error={setupError}
        forceConfigureStep={isReconfiguringSetup}
        onReturnToWorkspace={isRuntimeReady ? handleReturnToWorkspace : undefined}
        onStart={handleStartSetup}
        onRetry={handleRetrySetup}
        onCancel={handleCancelSetup}
      />
    );
  }

  return (
    <div className="app-shell">
      <DocumentsPane
        documents={documents}
        isFetchingDocuments={isFetchingDocuments}
        selectedDocIds={selectedDocIds}
        error={error}
        deletingDocId={deletingDocId}
        openMenuDocId={openMenuDocId}
        onSelectDocument={setSelectedDocIds}
        onToggleSelectAll={() =>
          setSelectedDocIds((current) =>
            current.length > 0 ? [] : documents.map((document) => document.doc_id),
          )
        }
        onDeleteDocument={handleDeleteDocument}
        onToggleMenu={setOpenMenuDocId}
      />

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
        onOpenSettings={handleOpenSetupSettings}
        onUploadFile={handleUploadFile}
      />
    </div>
  );
}
