import { useEffect, useState } from "react";
import type {
  GeneratorLoadPreset,
  SetupOptions,
  SetupProgressItem,
  SetupStatus,
  SetupStartPayload,
  SetupOption,
} from "../types";

type SetupPaneProps = {
  options: SetupOptions | null;
  status: SetupStatus | null;
  error: string | null;
  onStart: (payload: SetupStartPayload) => Promise<void>;
  onRetry: () => Promise<void>;
  onCancel: () => Promise<void>;
};

function generatorRuntimeGuide(
  preset: GeneratorLoadPreset,
  options: SetupOptions | null,
): { vramRange: string; recommendation: string } {
  const gpuName = options?.compute.gpu_name;
  const gpuMemory = options?.compute.gpu_memory_gb;
  const isRecommended = options?.compute.recommended_generator_load_preset === preset.key;
  const detectedGpu = gpuName
    ? gpuMemory
      ? `${gpuName} with ${gpuMemory} GB VRAM`
      : gpuName
    : null;

  switch (preset.key) {
    case "standard":
      return {
        vramRange: "Best on 10 GB+ VRAM",
        recommendation: isRecommended
          ? `Recommended for ${detectedGpu ?? "your detected GPU"}. Highest quality and simplest loading path.`
          : "Highest quality and simplest loading path when your GPU has comfortable VRAM headroom.",
      };
    case "bnb_8bit":
      return {
        vramRange: "Good fit for 7 to 9 GB VRAM",
        recommendation: isRecommended
          ? `Recommended for ${detectedGpu ?? "your detected GPU"}. Good balance between memory usage and quality.`
          : "Strong default for mid-range NVIDIA GPUs that need lower VRAM usage without a big quality drop.",
      };
    case "bnb_4bit":
      return {
        vramRange: "Best fit for 4 to 6 GB VRAM",
        recommendation: isRecommended
          ? `Recommended for ${detectedGpu ?? "your detected GPU"}. Lowest VRAM path on CUDA, with some quality tradeoff.`
          : "Use this for smaller NVIDIA GPUs where the standard loader may not fit in memory.",
      };
    case "cpu_safe":
      return {
        vramRange: "GPU VRAM not required",
        recommendation: "Use this when CUDA is unavailable or the GPU is too small.",
      };
    default:
      return {
        vramRange: preset.memory_hint,
        recommendation: preset.description,
      };
  }
}

function findOptionByKey<T extends { key: string }>(options: T[], key: string | null | undefined) {
  if (!key) {
    return null;
  }

  return options.find((option) => option.key === key) ?? null;
}

function formatSetupOptionLabel(option: SetupOption) {
  return option.size_hint ? `${option.label} · ${option.size_hint}` : option.label;
}

function formatProgressStateLabel(item: SetupProgressItem) {
  switch (item.status) {
    case "running":
      return "Downloading";
    case "complete":
      return "Ready";
    case "skipped":
      return "Already local";
    case "failed":
      return "Failed";
    default:
      return "Pending";
  }
}

function SetupSelectField({
  label,
  value,
  onChange,
  items,
  helper,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  items: Array<{ value: string; label: string }>;
  helper: string;
}) {
  return (
    <label className="setup-select-field">
      <span className="setup-select-label">{label}</span>
      <select
        className="setup-select-input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {items.map((item) => (
          <option key={item.value} value={item.value}>
            {item.label}
          </option>
        ))}
      </select>
      <span className="setup-select-helper">{helper}</span>
    </label>
  );
}

export function SetupPane({
  options,
  status,
  error,
  onStart,
  onRetry,
  onCancel,
}: SetupPaneProps) {
  const [generatorKey, setGeneratorKey] = useState<string>("");
  const [embeddingKey, setEmbeddingKey] = useState<string>("");
  const [generatorLoadPreset, setGeneratorLoadPreset] = useState<string>("");
  const [torchVariant, setTorchVariant] = useState<string>("");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!options) {
      return;
    }

    setGeneratorKey((current) => {
      if (current && options.generator_models.some((option) => option.key === current)) {
        return current;
      }
      return status?.selected_generator_key ?? options.generator_models[0]?.key ?? "";
    });

    setEmbeddingKey((current) => {
      if (current && options.embedding_models.some((option) => option.key === current)) {
        return current;
      }
      return status?.selected_embedding_key ?? options.embedding_models[0]?.key ?? "";
    });

    setGeneratorLoadPreset((current) => {
      if (current && options.generator_load_presets.some((option) => option.key === current)) {
        return current;
      }
      return (
        status?.selected_generator_load_preset ??
        options.compute.recommended_generator_load_preset ??
        options.generator_load_presets[0]?.key ??
        ""
      );
    });

    setTorchVariant((current) => {
      if (current && options.torch_variants.some((option) => option.key === current)) {
        return current;
      }
      return (
        status?.selected_torch_variant ??
        options.compute.recommended_torch_variant ??
        options.torch_variants[0]?.key ??
        ""
      );
    });
  }, [options, status]);

  async function handleStart() {
    if (!generatorKey || !embeddingKey || !generatorLoadPreset || !torchVariant) {
      setSubmitError("Select a generator, embedder, runtime preset, and torch variant.");
      return;
    }

    setSubmitError(null);
    setIsSubmitting(true);
    try {
      await onStart({
        generator_key: generatorKey,
        embedding_key: embeddingKey,
        generator_load_preset: generatorLoadPreset,
        torch_variant: torchVariant,
      });
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to start setup.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleRetry() {
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      await onRetry();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to retry setup.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleCancel() {
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      await onCancel();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to cancel setup.");
    } finally {
      setIsSubmitting(false);
    }
  }

  const isBusy = status?.is_busy ?? false;
  const generatorSelection = findOptionByKey(options?.generator_models ?? [], generatorKey);
  const embeddingSelection = findOptionByKey(options?.embedding_models ?? [], embeddingKey);
  const presetSelection = findOptionByKey(options?.generator_load_presets ?? [], generatorLoadPreset);
  const torchSelection = findOptionByKey(options?.torch_variants ?? [], torchVariant);
  const presetGuide = presetSelection ? generatorRuntimeGuide(presetSelection, options) : null;
  const detectedGpuLabel = options?.compute.cuda_available
    ? options.compute.gpu_name
      ? options.compute.gpu_memory_gb
        ? `${options.compute.gpu_name} · ${options.compute.gpu_memory_gb} GB VRAM`
        : options.compute.gpu_name
      : "NVIDIA GPU detected"
    : "CPU-only system";
  const shouldShowProgress =
    Boolean(status?.current_step) ||
    Boolean(status?.model_progress_items.length) ||
    status?.install_state === "ready" ||
    status?.install_state === "failed";

  return (
    <div className="setup-shell">
      <section className="setup-hero">
        <p className="setup-kicker">Desktop bootstrap setup</p>
        <h1>Prepare the local runtime before opening the workspace.</h1>
        <p className="setup-copy">
          Choose the local generator, embedding model, and runtime path. The installer
          creates the managed environment under
          <code>%LOCALAPPDATA%\LocalDocumentIntelligence</code> and downloads only what
          this machine needs.
        </p>

        <div className="setup-status-banner">
          <span className="setup-status-label">
            {status?.install_state === "ready"
              ? "Ready"
              : status?.install_state === "failed"
                ? "Setup failed"
                : status?.install_state === "installing"
                  ? "Installing"
                  : "Setup required"}
          </span>
          <span className="setup-status-message">
            {status?.progress_message ?? "Choose your runtime and start setup."}
          </span>
        </div>

        {shouldShowProgress && (
          <div className="setup-progress-board">
            <div className="setup-progress-card">
              <div className="setup-progress-heading">
                <span>Overall setup</span>
                <strong>{status?.overall_progress ?? 0}%</strong>
              </div>
              <div className="setup-progress-track overall">
                <div
                  className="setup-progress-fill overall"
                  style={{ width: `${status?.overall_progress ?? 0}%` }}
                />
              </div>
              <p className="setup-progress-caption">
                {status?.progress_message ?? "Waiting for setup to start."}
              </p>
            </div>

            <div className="setup-progress-card">
              <div className="setup-progress-heading">
                <span>Packages and runtime</span>
                <strong>{status?.package_progress ?? 0}%</strong>
              </div>
              <div className="setup-progress-track package">
                <div
                  className="setup-progress-fill package"
                  style={{ width: `${status?.package_progress ?? 0}%` }}
                />
              </div>
              <p className="setup-progress-caption">
                {status?.package_message ?? "The managed runtime has not started yet."}
              </p>
            </div>

            {Boolean(status?.model_progress_items.length) && (
              <div className="setup-model-progress-list">
                {status?.model_progress_items.map((item) => (
                  <div key={item.key} className="setup-model-progress-row">
                    <div className="setup-model-progress-header">
                      <span>{item.label}</span>
                      <span className={`setup-model-progress-state ${item.status}`}>
                        {formatProgressStateLabel(item)}
                      </span>
                    </div>
                    <div
                      className={`setup-progress-track item ${
                        item.status === "running" ? "is-running" : ""
                      }`}
                    >
                      <div
                        className={`setup-progress-fill item ${item.status}`}
                        style={{ width: `${item.progress}%` }}
                      />
                    </div>
                    <p className="setup-model-progress-detail">
                      {item.detail ?? "Waiting for this asset to be processed."}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="setup-hardware-strip">
          <div className="setup-hardware-item">
            <span className="setup-hardware-label">Detected hardware</span>
            <span className="setup-hardware-value">{detectedGpuLabel}</span>
          </div>
          <div className="setup-hardware-item">
            <span className="setup-hardware-label">Suggested runtime</span>
            <span className="setup-hardware-value">
              {presetSelection?.label ?? "Select a runtime preset"}
            </span>
          </div>
        </div>

        {(error || submitError || status?.last_error) && (
          <p className="query-error">
            {submitError ?? error ?? status?.last_error ?? "Unknown setup error."}
          </p>
        )}
      </section>

      <section className="setup-grid">
        <div className="setup-panel">
          <div className="setup-panel-header">
            <h2>Generator model</h2>
            <p>Model used for grounded answers and chat.</p>
          </div>
          <SetupSelectField
            label="Generator"
            value={generatorKey}
            onChange={setGeneratorKey}
            items={(options?.generator_models ?? []).map((option) => ({
              value: option.key,
              label: formatSetupOptionLabel(option),
            }))}
            helper={generatorSelection?.description ?? generatorSelection?.repo_id ?? "Select a local generator."}
          />
        </div>

        <div className="setup-panel">
          <div className="setup-panel-header">
            <h2>Embedding model</h2>
            <p>Retriever used for local document search.</p>
          </div>
          <SetupSelectField
            label="Embedder"
            value={embeddingKey}
            onChange={setEmbeddingKey}
            items={(options?.embedding_models ?? []).map((option) => ({
              value: option.key,
              label: formatSetupOptionLabel(option),
            }))}
            helper={embeddingSelection?.description ?? embeddingSelection?.repo_id ?? "Select a retrieval embedding model."}
          />
        </div>

        <div className="setup-panel">
          <div className="setup-panel-header">
            <h2>Generator runtime</h2>
            <p>Pick the memory profile that best matches the available VRAM.</p>
          </div>
          <SetupSelectField
            label="Runtime preset"
            value={generatorLoadPreset}
            onChange={setGeneratorLoadPreset}
            items={(options?.generator_load_presets ?? []).map((preset) => {
              const guide = generatorRuntimeGuide(preset, options);
              return {
                value: preset.key,
                label: `${preset.label} · ${guide.vramRange}`,
              };
            })}
            helper={presetGuide ? `${presetGuide.recommendation} ${presetSelection?.description ?? ""}` : "Select a runtime preset."}
          />
        </div>

        <div className="setup-panel">
          <div className="setup-panel-header">
            <h2>PyTorch runtime</h2>
            <p>Install the backend runtime for CPU or NVIDIA CUDA.</p>
          </div>
          <SetupSelectField
            label="PyTorch build"
            value={torchVariant}
            onChange={setTorchVariant}
            items={(options?.torch_variants ?? []).map((variant) => ({
              value: variant.key,
              label: variant.label,
            }))}
            helper={torchSelection?.description ?? "Select the runtime backend."}
          />
        </div>
      </section>

      <section className="setup-summary-panel">
        <div className="setup-summary-copy">
          <h2>Summary</h2>
          <p>
            {generatorSelection?.label ?? "Generator not selected"} ·{" "}
            {embeddingSelection?.label ?? "Embedder not selected"} ·{" "}
            {presetSelection?.label ?? "Preset not selected"} ·{" "}
            {torchSelection?.label ?? "Torch runtime not selected"}
          </p>
        </div>

        <div className="setup-actions">
          {status?.install_state === "failed" && (
            <button
              type="button"
              className="setup-action-button secondary"
              onClick={() => void handleRetry()}
              disabled={isBusy || isSubmitting}
            >
              Retry
            </button>
          )}

          {isBusy ? (
            <button
              type="button"
              className="setup-action-button danger"
              onClick={() => void handleCancel()}
              disabled={isSubmitting}
            >
              Cancel setup
            </button>
          ) : (
            <button
              type="button"
              className="setup-action-button primary"
              onClick={() => void handleStart()}
              disabled={!options || isSubmitting}
            >
              Start setup
            </button>
          )}
        </div>
      </section>
    </div>
  );
}
