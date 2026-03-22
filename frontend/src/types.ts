export type DocumentItem = {
  doc_id: string;
  original_filename: string;
};

export type DocumentsResponse = {
  items: DocumentItem[];
};

export type DocumentUploadResponse = {
  message: string;
  deduplicated: boolean;
  document: DocumentItem;
};

export type DocumentDeleteResponse = {
  message: string;
  doc_id: string;
};

export type UiQueryMode = "corpus" | "document" | "chat" | "auto";

export type BackendQueryMode = "grounded" | "chat" | "auto";

export type QueryRequestPayload = {
  query: string;
  mode: BackendQueryMode;
  doc_ids?: string[];
};

export type QuerySource = {
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

export type QueryStreamStart = {
  query: string;
  mode_used: string;
  fallback_reason: string | null;
  sources: QuerySource[];
  used_context_tokens: number;
  retrieved_chunk_count: number;
  retrieval_seconds: number;
};

export type QueryStreamDone = {
  answer: string;
  timings: {
    retrieval_seconds: number;
    generation_seconds: number;
    total_seconds: number;
  };
};

export type QueryStreamEvent =
  | { type: "start"; data: QueryStreamStart }
  | { type: "token"; data: { text: string } }
  | { type: "done"; data: QueryStreamDone }
  | { type: "error"; data: { message: string } };

export type QueryStatus = "idle" | "retrieving" | "generating" | "error";
