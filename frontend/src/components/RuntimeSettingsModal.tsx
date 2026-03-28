import { useEffect, useMemo, useState } from "react";
import type { GeneratorLoadPreset, SetupOption, SetupOptions, SetupStatus } from "../types";

type RuntimeSettingsModalProps = {
  options: SetupOptions;
  status: SetupStatus;
  error: string | null;
  isApplying: boolean;
  isQueryBusy: boolean;
  onClose: () => void;
  onApply: (payload: {
    generator_key: string;
    generator_load_preset: string;
  }) => Promise<void>;
};

function describeModel(option: SetupOption | null) {
  if (!option) {
    return "Select a generator model.";
  }

  return option.description ?? option.repo_id ?? "Select a generator model.";
}

function describePreset(option: GeneratorLoadPreset | null) {
  if (!option) {
    return "Select how the runtime should load the generator.";
  }

  return `${option.description} ${option.memory_hint}`.trim();
}

export function RuntimeSettingsModal({
  options,
  status,
  error,
  isApplying,
  isQueryBusy,
  onClose,
  onApply,
}: RuntimeSettingsModalProps) {
  const currentGeneratorKey = status.selected_generator_key ?? "";
  const currentPresetKey = status.selected_generator_load_preset ?? "";
  const torchVariantKey = status.selected_torch_variant ?? "cpu";
  const isCpuTorchRuntime = torchVariantKey === "cpu";
  const [selectedGeneratorKey, setSelectedGeneratorKey] = useState(currentGeneratorKey);
  const [selectedPresetKey, setSelectedPresetKey] = useState(currentPresetKey);

  useEffect(() => {
    setSelectedGeneratorKey(currentGeneratorKey);
  }, [currentGeneratorKey]);

  useEffect(() => {
    setSelectedPresetKey(currentPresetKey);
  }, [currentPresetKey]);

  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !isApplying) {
        onClose();
      }
    }

    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [isApplying, onClose]);

  const currentGenerator = useMemo(
    () =>
      options.generator_models.find((option) => option.key === currentGeneratorKey) ?? null,
    [currentGeneratorKey, options.generator_models],
  );
  const currentPreset = useMemo(
    () =>
      options.generator_load_presets.find((option) => option.key === currentPresetKey) ?? null,
    [currentPresetKey, options.generator_load_presets],
  );
  const selectedGenerator = useMemo(
    () =>
      options.generator_models.find((option) => option.key === selectedGeneratorKey) ?? null,
    [options.generator_models, selectedGeneratorKey],
  );
  const selectedPreset = useMemo(
    () =>
      options.generator_load_presets.find((option) => option.key === selectedPresetKey) ?? null,
    [options.generator_load_presets, selectedPresetKey],
  );

  const hasUnavailableGenerators = options.generator_models.some(
    (option) => option.is_downloaded === false,
  );
  const generatorSelectionBlocked = selectedGenerator?.is_downloaded === false;
  const presetSelectionBlocked =
    isCpuTorchRuntime &&
    (selectedPreset?.key === "bnb_8bit" || selectedPreset?.key === "bnb_4bit");
  const isSelectionUnchanged =
    selectedGeneratorKey === currentGeneratorKey &&
    selectedPresetKey === currentPresetKey;
  const isApplyDisabled =
    isApplying ||
    isQueryBusy ||
    isSelectionUnchanged ||
    !selectedGenerator ||
    !selectedPreset ||
    generatorSelectionBlocked ||
    presetSelectionBlocked;

  return (
    <div
      className="runtime-settings-overlay"
      onClick={(event) => {
        if (event.target === event.currentTarget && !isApplying) {
          onClose();
        }
      }}
    >
      <section
        className="runtime-settings-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="runtime-settings-title"
      >
        <div className="runtime-settings-header">
          <div className="runtime-settings-header-copy">
            <p className="runtime-settings-kicker">Runtime settings</p>
            <h2 id="runtime-settings-title">Switch the active generator without rerunning setup.</h2>
            <p className="runtime-settings-copy">
              This updates the live runtime only. The embedder and PyTorch build stay the same.
            </p>
          </div>

          <button
            type="button"
            className="runtime-settings-close"
            onClick={onClose}
            disabled={isApplying}
            aria-label="Close runtime settings"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6l12 12" />
              <path d="M18 6l-12 12" />
            </svg>
          </button>
        </div>

        <div className="runtime-settings-summary-grid">
          <article className="runtime-settings-summary-card">
            <span className="runtime-settings-summary-label">Current generator</span>
            <strong className="runtime-settings-summary-value">
              {currentGenerator?.label ?? "Not configured"}
            </strong>
            <p className="runtime-settings-summary-detail">
              {describeModel(currentGenerator)}
            </p>
          </article>

          <article className="runtime-settings-summary-card">
            <span className="runtime-settings-summary-label">Current runtime preset</span>
            <strong className="runtime-settings-summary-value">
              {currentPreset?.label ?? "Not configured"}
            </strong>
            <p className="runtime-settings-summary-detail">
              {describePreset(currentPreset)}
            </p>
          </article>

          <article className="runtime-settings-summary-card">
            <span className="runtime-settings-summary-label">Installed PyTorch runtime</span>
            <strong className="runtime-settings-summary-value">
              {options.torch_variants.find((variant) => variant.key === torchVariantKey)?.label ??
                torchVariantKey.toUpperCase()}
            </strong>
            <p className="runtime-settings-summary-detail">
              {isCpuTorchRuntime
                ? "Bitsandbytes presets stay unavailable on the CPU build."
                : "CUDA is available, so low-memory bitsandbytes presets can stay enabled."}
            </p>
          </article>
        </div>

        {error && <p className="query-error runtime-settings-error">{error}</p>}

        <div className="runtime-settings-fields">
          <label className="runtime-settings-field">
            <span className="runtime-settings-label">Generator model</span>
            <select
              className="runtime-settings-select"
              value={selectedGeneratorKey}
              onChange={(event) => setSelectedGeneratorKey(event.target.value)}
              disabled={isApplying}
            >
              {options.generator_models.map((option) => (
                <option
                  key={option.key}
                  value={option.key}
                  disabled={option.is_downloaded === false}
                >
                  {option.size_hint ? `${option.label} · ${option.size_hint}` : option.label}
                  {option.is_downloaded === false ? " · Install through setup" : ""}
                </option>
              ))}
            </select>
            <p className="runtime-settings-helper">
              {describeModel(selectedGenerator)}
            </p>
          </label>

          <label className="runtime-settings-field">
            <span className="runtime-settings-label">Generator runtime preset</span>
            <select
              className="runtime-settings-select"
              value={selectedPresetKey}
              onChange={(event) => setSelectedPresetKey(event.target.value)}
              disabled={isApplying}
            >
              {options.generator_load_presets.map((preset) => {
                const isDisabled =
                  isCpuTorchRuntime &&
                  (preset.key === "bnb_8bit" || preset.key === "bnb_4bit");

                return (
                  <option key={preset.key} value={preset.key} disabled={isDisabled}>
                    {preset.label}
                    {isDisabled ? " · Requires CUDA setup" : ""}
                  </option>
                );
              })}
            </select>
            <p className="runtime-settings-helper">
              {describePreset(selectedPreset)}
            </p>
          </label>
        </div>

        <div className="runtime-settings-notes">
          {hasUnavailableGenerators && (
            <p className="runtime-settings-note">
              Only generators that are already installed can be applied here. Use setup to
              install additional models first.
            </p>
          )}

          {isCpuTorchRuntime && (
            <p className="runtime-settings-note">
              Bitsandbytes presets require rerunning setup with a CUDA PyTorch runtime.
            </p>
          )}

          {isQueryBusy && (
            <p className="runtime-settings-note">
              Wait for the current query to finish before reloading the runtime.
            </p>
          )}
        </div>

        <div className="runtime-settings-actions">
          <button
            type="button"
            className="runtime-settings-button secondary"
            onClick={onClose}
            disabled={isApplying}
          >
            Cancel
          </button>

          <button
            type="button"
            className="runtime-settings-button primary"
            onClick={() =>
              void onApply({
                generator_key: selectedGeneratorKey,
                generator_load_preset: selectedPresetKey,
              })
            }
            disabled={isApplyDisabled}
          >
            {isApplying ? "Applying..." : "Apply and reload runtime"}
          </button>
        </div>
      </section>
    </div>
  );
}
