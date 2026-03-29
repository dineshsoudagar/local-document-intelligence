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
  forceConfigureStep?: boolean;
  onReturnToWorkspace?: () => void;
  onStart: (payload: SetupStartPayload) => Promise<void>;
  onRetry: () => Promise<void>;
  onCancel: () => Promise<void>;
};

type SetupStep = "configure" | "review";

type CurrentProgress = {
  activity: string;
  progress: number;
  stateClass: string;
  stateLabel: string;
  variant: "package" | "item";
};

function generatorRuntimeGuide(
  preset: GeneratorLoadPreset,
  options: SetupOptions | null,
): { vramRange: string; recommendation: string } {
  const gpuName = options?.compute.gpu_name;
  const gpuMemory = options?.compute.gpu_memory_gb;
  const isRecommended =
    options?.compute.recommended_generator_load_preset === preset.key;
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
        recommendation:
          "Use this when CUDA is unavailable or the GPU is too small.",
      };
    default:
      return {
        vramRange: preset.memory_hint,
        recommendation: preset.description,
      };
  }
}

function findOptionByKey<T extends { key: string }>(
  options: T[],
  key: string | null | undefined,
) {
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

function hasSetupHistory(status: SetupStatus | null) {
  if (!status || status.current_step === "cancelled") {
    return false;
  }

  if (status.current_step) {
    return true;
  }

  return status.model_progress_items.some(
    (item) => item.status !== "pending" || item.progress > 0,
  );
}

function shouldOpenReviewStep(status: SetupStatus | null) {
  if (!status) {
    return false;
  }

  if (
    status.install_state === "installing" ||
    status.install_state === "failed" ||
    status.install_state === "ready"
  ) {
    return true;
  }

  return hasSetupHistory(status);
}

function buildCurrentProgress(status: SetupStatus | null): CurrentProgress | null {
  if (!status) {
    return null;
  }

  const items = status.model_progress_items ?? [];
  if (status.current_step === "download_models" && items.length > 0) {
    const activeItem =
      items.find((item) => item.status === "running") ??
      items.find((item) => item.status === "pending") ??
      items.find(
        (item) => item.status === "complete" || item.status === "skipped",
      ) ??
      items[0];

    const completedCount = items.filter(
      (item) => item.status === "complete" || item.status === "skipped",
    ).length;
    const ordinal =
      activeItem.status === "pending"
        ? Math.min(completedCount + 1, items.length)
        : Math.max(completedCount, 1);
    const prefix =
      items.length > 1
        ? `Download ${ordinal} of ${items.length}`
        : "Model download";

    return {
      variant: "item",
      activity: `${prefix}: ${activeItem.label}`,
      progress: activeItem.progress,
      stateClass: activeItem.status,
      stateLabel: formatProgressStateLabel(activeItem),
    };
  }

  const packageState =
    status.install_state === "failed"
      ? "failed"
      : status.install_state === "ready"
        ? "complete"
        : "running";

  return {
    variant: "package",
    activity:
      status.package_message ??
      status.progress_message ??
      "Waiting for setup to start.",
    progress: status.package_progress ?? 0,
    stateClass: packageState,
    stateLabel:
      packageState === "failed"
        ? "Failed"
        : packageState === "complete"
          ? "Ready"
          : "Installing",
  };
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
  forceConfigureStep = false,
  onReturnToWorkspace,
  onStart,
  onRetry,
  onCancel,
}: SetupPaneProps) {
  const [generatorKey, setGeneratorKey] = useState<string>("");
  const [embeddingKey, setEmbeddingKey] = useState<string>("");
  const [generatorLoadPreset, setGeneratorLoadPreset] = useState<string>("");
  const [torchVariant, setTorchVariant] = useState<string>("");
  const [setupStep, setSetupStep] = useState<SetupStep>("configure");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!options) {
      return;
    }

    setGeneratorKey((current) => {
      if (
        current &&
        options.generator_models.some((option) => option.key === current)
      ) {
        return current;
      }
      return status?.selected_generator_key ?? options.generator_models[0]?.key ?? "";
    });

    setEmbeddingKey((current) => {
      if (
        current &&
        options.embedding_models.some((option) => option.key === current)
      ) {
        return current;
      }
      return status?.selected_embedding_key ?? options.embedding_models[0]?.key ?? "";
    });

    setGeneratorLoadPreset((current) => {
      if (
        current &&
        options.generator_load_presets.some((option) => option.key === current)
      ) {
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
      if (
        current &&
        options.torch_variants.some((option) => option.key === current)
      ) {
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

  useEffect(() => {
    if (forceConfigureStep) {
      setSetupStep("configure");
    }
  }, [forceConfigureStep]);

  useEffect(() => {
    if (forceConfigureStep) {
      return;
    }

    if (!status) {
      return;
    }

    if (status.current_step === "cancelled") {
      setSetupStep("configure");
      return;
    }

    if (shouldOpenReviewStep(status)) {
      setSetupStep("review");
    }
  }, [forceConfigureStep, status]);

  function validateSelections() {
    if (!generatorKey || !embeddingKey || !generatorLoadPreset || !torchVariant) {
      setSubmitError(
        "Select a generator, embedder, runtime preset, and torch variant.",
      );
      return false;
    }

    return true;
  }

  function handleNext() {
    if (!validateSelections()) {
      return;
    }

    setSubmitError(null);
    setSetupStep("review");
  }

  function handleBack() {
    setSubmitError(null);
    setSetupStep("configure");
  }

  async function handleStart() {
    if (!validateSelections()) {
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
  const generatorSelection = findOptionByKey(
    options?.generator_models ?? [],
    generatorKey,
  );
  const embeddingSelection = findOptionByKey(
    options?.embedding_models ?? [],
    embeddingKey,
  );
  const presetSelection = findOptionByKey(
    options?.generator_load_presets ?? [],
    generatorLoadPreset,
  );
  const torchSelection = findOptionByKey(
    options?.torch_variants ?? [],
    torchVariant,
  );
  const presetGuide = presetSelection
    ? generatorRuntimeGuide(presetSelection, options)
    : null;
  const detectedGpuLabel = options?.compute.cuda_available
    ? options.compute.gpu_name
      ? options.compute.gpu_memory_gb
        ? `${options.compute.gpu_name} · ${options.compute.gpu_memory_gb} GB VRAM`
        : options.compute.gpu_name
      : "NVIDIA GPU detected"
    : "CPU-only system";
  const isSelectionSyncedWithStatus = Boolean(
    status &&
      status.selected_generator_key === generatorKey &&
      status.selected_embedding_key === embeddingKey &&
      status.selected_generator_load_preset === generatorLoadPreset &&
      status.selected_torch_variant === torchVariant,
  );
  const canRetry =
    setupStep === "review" &&
    status?.install_state === "failed" &&
    isSelectionSyncedWithStatus;
  const reviewingPersistedRun =
    isSelectionSyncedWithStatus &&
    (status?.install_state === "installing" ||
      status?.install_state === "failed" ||
      hasSetupHistory(status));
  const shouldShowProgress = setupStep === "review" && reviewingPersistedRun;
  const currentProgress = buildCurrentProgress(status);
  const statusLabel =
    status?.install_state === "ready"
      ? "Ready"
      : status?.install_state === "installing"
        ? "Installing"
        : canRetry
          ? "Setup failed"
          : setupStep === "review"
            ? "Review setup"
            : "Setup required";
  const statusMessage =
    status?.install_state === "installing"
      ? status?.progress_message ?? "Installing the managed runtime..."
      : canRetry
        ? status?.last_error ??
          status?.progress_message ??
          "Setup failed. Review the summary or retry the last selection."
        : setupStep === "review"
          ? "Review the selected models and runtime, then start setup."
          : "Choose the local models and runtime, then continue to review.";
  const visibleError =
    submitError ??
    error ??
    (canRetry ? status?.last_error ?? null : null);
  const reviewSummaryItems = [
    {
      label: "Generator model",
      value: generatorSelection?.label ?? "Generator not selected",
      detail:
        generatorSelection?.description ??
        generatorSelection?.repo_id ??
        "Model used for grounded answers and chat.",
    },
    {
      label: "Embedding model",
      value: embeddingSelection?.label ?? "Embedder not selected",
      detail:
        embeddingSelection?.description ??
        embeddingSelection?.repo_id ??
        "Retriever used for local document search.",
    },
    {
      label: "Generator runtime",
      value: presetSelection?.label ?? "Preset not selected",
      detail:
        presetGuide?.recommendation ??
        presetSelection?.description ??
        "Memory profile for the selected generator.",
    },
    {
      label: "PyTorch runtime",
      value: torchSelection?.label ?? "Torch runtime not selected",
      detail:
        torchSelection?.description ??
        "Backend runtime for CPU or NVIDIA CUDA.",
    },
    {
      label: "Detected hardware",
      value: detectedGpuLabel,
      detail:
        options?.compute.cuda_available
          ? "Setup will install the runtime that fits this GPU."
          : "The managed runtime will use the CPU-safe path.",
    },
    {
      label: "Suggested runtime",
      value: presetSelection?.label ?? "Select a runtime preset",
      detail:
        presetGuide?.vramRange ??
        "Recommended from the detected hardware profile.",
    },
  ];

  return (
    <div className="setup-shell">
      <section className="setup-hero">
        <div className="setup-hero-top">
          <div className="setup-hero-copy">
            <p className="setup-kicker">Desktop bootstrap setup</p>
            <h1>
              {setupStep === "configure"
                ? "Choose the local runtime before opening the workspace."
                : "Review the setup before installing the local runtime."}
            </h1>
            <p className="setup-copy">
              The installer creates the managed environment under
              <code>%LOCALAPPDATA%\LocalDocumentIntelligence</code> and downloads
              only what this machine needs.
            </p>
          </div>

          <div className="setup-stepper" aria-label="Setup steps">
            <div
              className={`setup-step ${setupStep === "configure" ? "is-active" : "is-complete"}`}
            >
              <span className="setup-step-index">1</span>
              <span className="setup-step-text">Select models and runtime</span>
            </div>
            <div className={`setup-step ${setupStep === "review" ? "is-active" : ""}`}>
              <span className="setup-step-index">2</span>
              <span className="setup-step-text">Review and start setup</span>
            </div>
          </div>
        </div>

        <div className="setup-status-banner">
          <span className="setup-status-label">{statusLabel}</span>
          <span className="setup-status-message">{statusMessage}</span>
        </div>

        {setupStep === "configure" && (
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
        )}

        {visibleError && <p className="query-error">{visibleError}</p>}
      </section>

      {setupStep === "configure" ? (
        <>
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
                helper={
                  generatorSelection?.description ??
                  generatorSelection?.repo_id ??
                  "Select a local generator."
                }
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
                helper={
                  embeddingSelection?.description ??
                  embeddingSelection?.repo_id ??
                  "Select a retrieval embedding model."
                }
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
                helper={
                  presetGuide
                    ? `${presetGuide.recommendation} ${presetSelection?.description ?? ""}`
                    : "Select a runtime preset."
                }
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
                helper={
                  torchSelection?.description ??
                  "Select the runtime backend."
                }
              />
            </div>
          </section>

          <section className="setup-footer-panel">
            <div className="setup-footer-copy">
              <h2>Continue to review</h2>
              <p>
                Confirm the selected models and runtime on the next screen before
                setup begins.
              </p>
            </div>

            <div className="setup-actions">
              {onReturnToWorkspace && (
                <button
                  type="button"
                  className="setup-action-button secondary"
                  onClick={onReturnToWorkspace}
                  disabled={isSubmitting}
                >
                  Back to workspace
                </button>
              )}
              <button
                type="button"
                className="setup-action-button primary"
                onClick={handleNext}
                disabled={!options || isSubmitting}
              >
                Next
              </button>
            </div>
          </section>
        </>
      ) : (
        <>
          <section className="setup-review-panel">
            <div className="setup-review-header">
              <h2>Selection summary</h2>
              <p>
                {canRetry
                  ? "Retry the last setup or go back and adjust the selection before starting again."
                  : "Everything is ready. Start setup when this summary looks right."}
              </p>
            </div>

            <div className="setup-review-grid">
              {reviewSummaryItems.map((item) => (
                <article key={item.label} className="setup-review-card">
                  <span className="setup-review-label">{item.label}</span>
                  <strong className="setup-review-value">{item.value}</strong>
                  <p className="setup-review-detail">{item.detail}</p>
                </article>
              ))}
            </div>
          </section>

          {shouldShowProgress && currentProgress && (
            <section className="setup-progress-panel">
              <div className="setup-progress-card">
                <div className="setup-progress-row">
                  <p className="setup-progress-activity">
                    {status?.progress_message ?? "Preparing setup..."}
                  </p>
                  <div className="setup-progress-meta">
                    <span className="setup-progress-tag">Overall</span>
                    <strong className="setup-progress-percent">
                      {status?.overall_progress ?? 0}%
                    </strong>
                  </div>
                </div>
                <div className="setup-progress-track overall">
                  <div
                    className="setup-progress-fill overall"
                    style={{ width: `${status?.overall_progress ?? 0}%` }}
                  />
                </div>
              </div>

              <div className="setup-progress-card">
                <div className="setup-progress-row">
                  <p className="setup-progress-activity">{currentProgress.activity}</p>
                  <div className="setup-progress-meta">
                    <span
                      className={`setup-model-progress-state ${currentProgress.stateClass}`}
                    >
                      {currentProgress.stateLabel}
                    </span>
                    <strong className="setup-progress-percent">
                      {currentProgress.progress}%
                    </strong>
                  </div>
                </div>
                <div
                  className={`setup-progress-track ${
                    currentProgress.variant === "item"
                      ? `item ${currentProgress.stateClass === "running" ? "is-running" : ""}`
                      : "package"
                  }`}
                >
                  <div
                    className={
                      currentProgress.variant === "item"
                        ? `setup-progress-fill item ${currentProgress.stateClass}`
                        : "setup-progress-fill package"
                    }
                    style={{ width: `${currentProgress.progress}%` }}
                  />
                </div>
              </div>
            </section>
          )}

          <section className="setup-footer-panel">
            <div className="setup-footer-copy">
              <h2>
                {isBusy
                  ? "Installing local runtime"
                  : canRetry
                    ? "Setup needs attention"
                    : "Ready to start setup"}
              </h2>
              <p>
                {isBusy
                  ? "The installer is creating the managed environment and downloading the selected assets."
                  : canRetry
                    ? "Use Retry for the last selection, or go back to change the setup choices."
                    : "Start setup to install packages, the selected runtime, and the local models."}
              </p>
            </div>

            <div className="setup-actions">
              {onReturnToWorkspace && !isBusy && (
                <button
                  type="button"
                  className="setup-action-button secondary"
                  onClick={onReturnToWorkspace}
                  disabled={isSubmitting}
                >
                  Back to workspace
                </button>
              )}
              {!isBusy && (
                <button
                  type="button"
                  className="setup-action-button secondary"
                  onClick={handleBack}
                  disabled={isSubmitting}
                >
                  Back
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
              ) : canRetry ? (
                <button
                  type="button"
                  className="setup-action-button primary"
                  onClick={() => void handleRetry()}
                  disabled={isSubmitting}
                >
                  Retry
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
        </>
      )}
    </div>
  );
}
