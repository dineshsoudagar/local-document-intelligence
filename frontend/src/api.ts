import type {
  DocumentDeleteResponse,
  DocumentsResponse,
  DocumentUploadResponse,
  QueryRequestPayload,
  QueryStreamDone,
  QueryStreamEvent,
  QueryStreamStart,
} from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

export async function fetchDocuments() {
  const response = await fetch(apiUrl("/documents"));

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
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
    throw new Error(`Upload failed: ${response.status}`);
  }

  const data: DocumentUploadResponse = await response.json();
  return data;
}

export async function deleteDocument(docId: string) {
  const response = await fetch(apiUrl(`/documents/${docId}`), {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(`Delete failed: ${response.status}`);
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
    throw new Error(`Query failed: ${response.status}`);
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
