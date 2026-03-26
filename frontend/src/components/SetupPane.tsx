import { useEffect, useState } from "react";
import type {
  GeneratorLoadPreset,
  SetupOptions,
  SetupStatus,
  SetupStartPayload,
  SetupOption,
  TorchVariant,
} from "../types";

type SetupPaneProps = {
  options: SetupOptions | null;
  status: SetupStatus | null;
  error: string | null;
  onStart: (payload: SetupStartPayload) => Promise<void>;
  onRetry: () => Promise<void>;
  onCancel: () => Promise<void>;
};

function SetupCardOption({
  title,
  body,
  isSelected,
  onClick,
}: {
  title: string;
  body: string;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={isSelected ? "setup-card-option selected" : "setup-card-option"}
      onClick={onClick}
    >
      <span className="setup-card-option-title">{title}</span>
      <span className="setup-card-option-body">{body}</span>
    </button>
  );
}

function findOptionByKey<T extends { key: string }>(options: T[], key: string | null | undefined) {
  if (!key) {
    return null;
  }

  return options.find((option) => option.key === key) ?? null;
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

  return (
    <div className="setup-shell">
      <section className="setup-hero">
        <p className="setup-kicker">Desktop bootstrap setup</p>
        <h1>Prepare the local runtime before opening the workspace.</h1>
        <p className="setup-copy">
          This first-run flow creates the managed environment under
          <code>%LOCALAPPDATA%\LocalDocumentIntelligence</code>, installs the selected
          runtime, and downloads the models you need.
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
            <p>Choose the local model used for grounded answers and chat.</p>
          </div>
          <div className="setup-option-grid">
            {(options?.generator_models ?? []).map((option: SetupOption) => (
              <SetupCardOption
                key={option.key}
                title={`${option.label}${option.size_hint ? ` · ${option.size_hint}` : ""}`}
                body={option.description ?? option.repo_id ?? option.key}
                isSelected={generatorKey === option.key}
                onClick={() => setGeneratorKey(option.key)}
              />
            ))}
          </div>
        </div>

        <div className="setup-panel">
          <div className="setup-panel-header">
            <h2>Embedding model</h2>
            <p>Choose the dense retriever that powers document search.</p>
          </div>
          <div className="setup-option-grid">
            {(options?.embedding_models ?? []).map((option: SetupOption) => (
              <SetupCardOption
                key={option.key}
                title={`${option.label}${option.size_hint ? ` · ${option.size_hint}` : ""}`}
                body={option.description ?? option.repo_id ?? option.key}
                isSelected={embeddingKey === option.key}
                onClick={() => setEmbeddingKey(option.key)}
              />
            ))}
          </div>
        </div>

        <div className="setup-panel">
          <div className="setup-panel-header">
            <h2>Generator runtime</h2>
            <p>
              Standard keeps maximum compatibility. Bitsandbytes presets lower VRAM use
              when CUDA is available.
            </p>
          </div>
          <div className="setup-option-grid">
            {(options?.generator_load_presets ?? []).map((preset: GeneratorLoadPreset) => (
              <SetupCardOption
                key={preset.key}
                title={preset.label}
                body={`${preset.description} ${preset.memory_hint}`}
                isSelected={generatorLoadPreset === preset.key}
                onClick={() => setGeneratorLoadPreset(preset.key)}
              />
            ))}
          </div>
        </div>

        <div className="setup-panel">
          <div className="setup-panel-header">
            <h2>PyTorch runtime</h2>
            <p>
              {options?.compute.cuda_available
                ? `Detected NVIDIA GPU: ${options.compute.gpu_name ?? "CUDA available"}.`
                : "No NVIDIA GPU detected. CPU runtime is recommended."}
            </p>
          </div>
          <div className="setup-option-grid">
            {(options?.torch_variants ?? []).map((variant: TorchVariant) => (
              <SetupCardOption
                key={variant.key}
                title={variant.label}
                body={variant.description}
                isSelected={torchVariant === variant.key}
                onClick={() => setTorchVariant(variant.key)}
              />
            ))}
          </div>
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
