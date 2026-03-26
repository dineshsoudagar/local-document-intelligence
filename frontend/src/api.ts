import type {
  DocumentDeleteResponse,
  DocumentsResponse,
  DocumentUploadResponse,
  QueryRequestPayload,
  QueryStreamDone,
  QueryStreamEvent,
  QueryStreamStart,
  SetupOptions,
  SetupStartPayload,
  SetupStatus,
} from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

async function readErrorMessage(response: Response, fallback: string) {
  try {
    const payload = await response.json();
    if (payload && typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    // Fall back to a generic message when the backend did not return JSON.
  }

  return `${fallback}: ${response.status}`;
}

export async function fetchDocuments() {
  const response = await fetch(apiUrl("/documents"));

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Request failed"));
  }

  const data: DocumentsResponse = await response.json();
  return data.items;
}

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(apiUrl("/documents/upload"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Upload failed"));
  }

  const data: DocumentUploadResponse = await response.json();
  return data;
}

export async function deleteDocument(docId: string) {
  const response = await fetch(apiUrl(`/documents/${docId}`), {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Delete failed"));
  }

  const data: DocumentDeleteResponse = await response.json();
  return data;
}

type StreamQueryHandlers = {
  onStart: (data: QueryStreamStart) => void;
  onToken: (text: string) => void;
  onDone: (data: QueryStreamDone) => void;
};

export async function streamQuery(
  payload: QueryRequestPayload,
  signal: AbortSignal,
  handlers: StreamQueryHandlers,
) {
  const response = await fetch(apiUrl("/query/stream"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Query failed"));
  }

  if (!response.body) {
    throw new Error("Streaming response body is missing.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function processEvent(event: QueryStreamEvent) {
    if (event.type === "start") {
      handlers.onStart(event.data);
      return;
    }

    if (event.type === "answer_token") {
      handlers.onToken(event.data.text);
      return;
    }

    if (event.type === "thinking_token" || event.type === "thinking_done") {
      return;
    }

    if (event.type === "done") {
      handlers.onDone(event.data);
      return;
    }

    if (event.type === "error") {
      throw new Error(event.data.message);
    }

    throw new Error(`Unsupported stream event type: ${(event as { type: string }).type}`);
  }

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();

      if (!trimmed) {
        continue;
      }

      processEvent(JSON.parse(trimmed) as QueryStreamEvent);
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    processEvent(JSON.parse(trailing) as QueryStreamEvent);
  }
}

export async function fetchSetupStatus() {
  const response = await fetch(apiUrl("/setup/status"));

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Failed to load setup status"));
  }

  return (await response.json()) as SetupStatus;
}

export async function fetchSetupOptions() {
  const response = await fetch(apiUrl("/setup/options"));

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Failed to load setup options"));
  }

  return (await response.json()) as SetupOptions;
}

export async function startSetup(payload: SetupStartPayload) {
  const response = await fetch(apiUrl("/setup/start"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Failed to start setup"));
  }

  return (await response.json()) as SetupStatus;
}

export async function retrySetup() {
  const response = await fetch(apiUrl("/setup/retry"), {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Failed to retry setup"));
  }

  return (await response.json()) as SetupStatus;
}

export async function cancelSetup() {
  const response = await fetch(apiUrl("/setup/cancel"), {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Failed to cancel setup"));
  }

  return (await response.json()) as SetupStatus;
}
