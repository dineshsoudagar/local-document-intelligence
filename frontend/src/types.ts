// Minimal document metadata returned by the backend and shown in the document list.
export type DocumentItem = {
  doc_id: string;
  original_filename: string;
};

// Wrapper used by the documents listing endpoint.
export type DocumentsResponse = {
  items: DocumentItem[];
};

// Response returned after a successful upload attempt.
export type DocumentUploadResponse = {
  message: string;
  deduplicated: boolean;
  document: DocumentItem;
};

// Response returned after deleting one uploaded document.
export type DocumentDeleteResponse = {
  message: string;
  doc_id: string;
};

// Query modes exposed by the frontend controls.
export type UiQueryMode = "corpus" | "document" | "chat" | "auto";

// Query modes accepted by the backend API.
export type BackendQueryMode = "grounded" | "chat" | "auto";

// One rendered chat message in the frontend transcript.
export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "complete" | "error";
};

// Payload sent when starting a new query request.
export type QueryRequestPayload = {
  query: string;
  mode: BackendQueryMode;
  doc_ids?: string[];
};

// One retrieved evidence chunk returned by the backend for grounding.
export type QuerySource = {
  doc_id?: string | null;
  original_filename?: string | null;
  pages: number[];
};

// First streaming event: metadata about retrieval before answer tokens begin.
export type QueryStreamStart = {
  query: string;
  mode_used: string;
  fallback_reason: string | null;
  sources: QuerySource[];
  used_context_tokens: number;
  retrieved_chunk_count: number;
  retrieval_seconds: number;
};

// Final streaming event: complete answer text and timing information.
export type QueryStreamDone = {
  answer: string;
  timings: {
    retrieval_seconds: number;
    generation_seconds: number;
    total_seconds: number;
  };
};

// All event shapes that can arrive over the streaming query endpoint.
export type QueryStreamEvent =
  | { type: "start"; data: QueryStreamStart }
  | { type: "token"; data: { text: string } }
  | { type: "done"; data: QueryStreamDone }
  | { type: "error"; data: { message: string } };

// Small UI state machine used to drive loading indicators and status text.
export type QueryStatus = "idle" | "retrieving" | "generating" | "error";
