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
  sources?: QuerySource[];
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
  thinking_content?: string | null;
  thinking_finished?: boolean;
  reasoning_mode?: string;
  timings: {
    retrieval_seconds: number;
    generation_seconds: number;
    total_seconds: number;
  };
};

// All event shapes that can arrive over the streaming query endpoint.
export type QueryStreamEvent =
  | { type: "start"; data: QueryStreamStart }
  | { type: "answer_token"; data: { text: string } }
  | { type: "thinking_token"; data: { text: string } }
  | { type: "thinking_done"; data: Record<string, never> }
  | { type: "done"; data: QueryStreamDone }
  | { type: "error"; data: { message: string } };

// Small UI state machine used to drive loading indicators and status text.
export type QueryStatus = "idle" | "retrieving" | "generating" | "error";

export type SetupOption = {
  key: string;
  role?: string | null;
  label: string;
  description?: string | null;
  size_hint?: string | null;
  repo_id?: string | null;
};

export type GeneratorLoadPreset = {
  key: string;
  label: string;
  description: string;
  memory_hint: string;
  generator_load_mode: "standard" | "bnb_8bit" | "bnb_4bit";
  generator_dtype: "auto" | "float16" | "bfloat16" | "float32";
  generator_device_map?: string | null;
  bnb_4bit_quant_type?: "nf4" | "fp4";
  bnb_4bit_use_double_quant?: boolean;
  bnb_int8_enable_fp32_cpu_offload?: boolean;
};

export type TorchVariant = {
  key: string;
  label: string;
  description: string;
  index_url: string;
};

export type SetupComputeInfo = {
  cuda_available: boolean;
  gpu_name?: string | null;
  gpu_memory_gb?: number | null;
  recommended_torch_variant: string;
  recommended_generator_load_preset: string;
  allowed_torch_variants: string[];
};

export type SetupOptions = {
  generator_models: SetupOption[];
  embedding_models: SetupOption[];
  generator_load_presets: GeneratorLoadPreset[];
  compute: SetupComputeInfo;
  torch_variants: TorchVariant[];
};

export type SetupStatus = {
  install_state: "not_ready" | "installing" | "ready" | "failed";
  current_step?: string | null;
  progress_message?: string | null;
  last_error?: string | null;
  cancel_requested: boolean;
  is_busy: boolean;
  selected_generator_key?: string | null;
  selected_embedding_key?: string | null;
  selected_generator_load_preset?: string | null;
  selected_torch_variant?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at: string;
};

export type SetupStartPayload = {
  generator_key: string;
  embedding_key: string;
  generator_load_preset: string;
  torch_variant: string;
};
