"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  FaCheckCircle,
  FaColumns,
  FaFileCsv,
  FaFlask,
  FaLayerGroup,
  FaPlay,
  FaUpload,
  FaWaveSquare,
} from "react-icons/fa";
import ProbabilityGauge from "@/components/charts/ProbabilityGauge";
import {
  ActionButton,
  EmptyState,
  PageIntro,
  Panel,
  SiteContainer,
  StatusPill,
  cx,
} from "@/components/site/ui";
import { usePrediction } from "@/context/PredictionContext";
import { parseCsvFeatureRow } from "@/lib/csv";
import {
  explain,
  fetchFeatures,
  fetchModelRegistry,
  predict,
  type FeatureGroup,
  type FeaturesData,
  type ModelRegistryData,
} from "@/lib/api";
import {
  confidenceLabel,
  friendlyFeatureName,
  modelDisplayName,
  predictionLabel,
  riskLabel,
  riskTone,
} from "@/lib/prediction-utils";

function buildNumericFeatures(
  rawValues: Record<string, string>,
  expectedFeatures: string[]
): Record<string, number> {
  const numericFeatures: Record<string, number> = {};

  for (const feature of expectedFeatures) {
    const rawValue = rawValues[feature]?.trim() ?? "";
    const parsed = rawValue === "" ? 0 : Number.parseFloat(rawValue);
    if (!Number.isFinite(parsed)) {
      throw new Error(`Feature \`${feature}\` must be numeric.`);
    }
    numericFeatures[feature] = parsed;
  }

  return numericFeatures;
}

function valuesFromSample(featuresInfo: FeaturesData): Record<string, string> {
  const nextValues: Record<string, string> = {};
  featuresInfo.expected_features.forEach((feature) => {
    nextValues[feature] =
      featuresInfo.sample_data[feature] !== undefined
        ? String(featuresInfo.sample_data[feature])
        : "";
  });
  return nextValues;
}

function fallbackGroups(featuresInfo: FeaturesData | null): FeatureGroup[] {
  if (!featuresInfo) {
    return [];
  }

  return featuresInfo.feature_groups?.length
    ? featuresInfo.feature_groups
    : [
        {
          name: "Selected voice features",
          description: "Exact deployed feature columns used by the saved models.",
          features: featuresInfo.expected_features,
        },
      ];
}

export default function UploadPage() {
  const {
    prediction,
    modelDisplayName: storedModelDisplayName,
    uploadedFileName,
    setPredictionBundle,
  } = usePrediction();

  const [featuresInfo, setFeaturesInfo] = useState<FeaturesData | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryData | null>(null);
  const [featureValues, setFeatureValues] = useState<Record<string, string>>({});
  const [selectedModel, setSelectedModel] = useState("xgboost");
  const [inputMode, setInputMode] = useState<"grouped" | "advanced">("grouped");
  const [fileName, setFileName] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    Promise.all([fetchFeatures(), fetchModelRegistry()])
      .then(([featuresData, registryData]) => {
        if (cancelled) {
          return;
        }

        setFeaturesInfo(featuresData);
        setRegistry(registryData);
        setSelectedModel(
          registryData.best_model || featuresData.supported_models[0] || "xgboost"
        );
        setFeatureValues(valuesFromSample(featuresData));
      })
      .catch((fetchError) => {
        if (!cancelled) {
          setError(
            fetchError instanceof Error
              ? fetchError.message
              : "Failed to load feature metadata."
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const groups = fallbackGroups(featuresInfo);
  const selectedDisplayName = modelDisplayName(selectedModel, registry?.models ?? []);
  const displayedPrediction = prediction;
  const displayedRisk = displayedPrediction
    ? riskLabel(displayedPrediction.probability)
    : null;
  const reviewModelName = storedModelDisplayName || selectedDisplayName;

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFeatureValues((current) => ({
      ...current,
      [event.target.name]: event.target.value,
    }));
  };

  const handleLoadSample = () => {
    if (!featuresInfo) {
      return;
    }

    setFeatureValues(valuesFromSample(featuresInfo));
    setNotice("Loaded the saved sample row from the training artifact.");
    setError("");
  };

  const runAnalysis = async (
    features: Record<string, number>,
    uploadedFileLabel: string | null = null
  ) => {
    if (!featuresInfo) {
      return;
    }

    setLoading(true);
    setError("");
    setPredictionBundle(null);

    try {
      const [predictionResult, explanationResult] = await Promise.all([
        predict(features, selectedModel),
        explain(features, selectedModel),
      ]);

      setPredictionBundle({
        prediction: predictionResult,
        explanation: explanationResult,
        features,
        model: selectedModel,
        modelDisplayName: selectedDisplayName,
        trainingDataset: featuresInfo.training_dataset,
        uploadedFileName: uploadedFileLabel,
      });
    } catch (analysisError) {
      setError(
        analysisError instanceof Error
          ? analysisError.message
          : "Prediction failed."
      );
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!featuresInfo) {
      return;
    }

    setNotice("");

    try {
      const numericFeatures = buildNumericFeatures(
        featureValues,
        featuresInfo.expected_features
      );
      await runAnalysis(numericFeatures, null);
    } catch (validationError) {
      setError(
        validationError instanceof Error
          ? validationError.message
          : "Invalid feature values."
      );
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file || !featuresInfo) {
      return;
    }

    setFileName(file.name);
    setError("");

    try {
      const csvText = await file.text();
      const { features, rowCount, ignoredColumns } = parseCsvFeatureRow(
        csvText,
        featuresInfo.expected_features
      );

      const nextValues: Record<string, string> = {};
      featuresInfo.expected_features.forEach((feature) => {
        nextValues[feature] = String(features[feature]);
      });
      setFeatureValues(nextValues);

      const noticeParts = [`Loaded ${file.name} into the analysis workspace.`];
      if (ignoredColumns.length > 0) {
        noticeParts.push(
          `Ignored ${ignoredColumns.length} extra column${
            ignoredColumns.length === 1 ? "" : "s"
          } not used by the deployed model.`
        );
      }
      if (rowCount > 1) {
        noticeParts.push("Used the first row for this single-sample prediction.");
      }
      setNotice(noticeParts.join(" "));

      await runAnalysis(features, file.name);
    } catch (csvError) {
      setError(
        csvError instanceof Error ? csvError.message : "CSV parsing failed."
      );
    } finally {
      event.target.value = "";
    }
  };

  return (
    <div className="pb-20">
      <PageIntro
        eyebrow="Analysis workspace"
        title="Run one registered voice model against one feature profile."
        description="Use grouped acoustic fields for a friendly demo or switch to exact-column mode for technical review. Every model shown here is trained on pd_speech_features.csv."
        meta={[
          {
            label: "Training dataset",
            value: featuresInfo?.training_dataset ?? "Loading",
          },
          {
            label: "Required features",
            value: featuresInfo ? `${featuresInfo.feature_count}` : "Loading",
          },
          {
            label: "Active model",
            value: selectedDisplayName,
          },
        ]}
      />

      <SiteContainer className="space-y-6 pt-10">
        {error ? (
          <div className="rounded-[24px] border border-[rgba(142,75,67,0.18)] bg-[rgba(142,75,67,0.1)] px-5 py-4 text-sm leading-7 text-[var(--danger)]">
            {error}
          </div>
        ) : null}

        {notice ? (
          <div className="rounded-[24px] border border-[rgba(29,72,80,0.14)] bg-[rgba(29,72,80,0.08)] px-5 py-4 text-sm leading-7 text-[var(--accent-strong)]">
            {notice}
          </div>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
          <div className="space-y-6">
            <Panel>
              <div className="grid gap-5 lg:grid-cols-[1fr_280px] lg:items-end">
                <div>
                  <p className="eyebrow">Model registry</p>
                  <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                    Choose the display-labeled model
                  </h2>
                  <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                    Names come from `backend/model_registry.json`; model keys stay stable for inference.
                  </p>
                </div>
                <div>
                  <label className="field-label" htmlFor="selected-model">
                    Active model
                  </label>
                  <select
                    id="selected-model"
                    value={selectedModel}
                    onChange={(event) => setSelectedModel(event.target.value)}
                    className="select-shell"
                  >
                    {(registry?.models ?? []).map((registeredModel) => (
                      <option key={registeredModel.key} value={registeredModel.key}>
                        {registeredModel.display_name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="mt-6 grid gap-4 md:grid-cols-2">
                {(registry?.models ?? []).map((registeredModel) => (
                  <button
                    key={registeredModel.key}
                    type="button"
                    onClick={() => setSelectedModel(registeredModel.key)}
                    className={cx(
                      "rounded-[26px] border p-5 text-left transition",
                      selectedModel === registeredModel.key
                        ? "border-[rgba(29,72,80,0.34)] bg-[rgba(29,72,80,0.1)]"
                        : "border-[var(--border-subtle)] bg-white/56 hover:bg-white/78"
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-lg font-semibold text-[var(--text-strong)]">
                          {registeredModel.display_name}
                        </p>
                        <p className="mt-2 text-sm leading-7 text-[var(--text-muted)]">
                          {registeredModel.description}
                        </p>
                      </div>
                      {selectedModel === registeredModel.key ? (
                        <FaCheckCircle className="mt-1 text-[var(--success)]" />
                      ) : null}
                    </div>
                    <p className="mt-4 text-xs font-bold uppercase tracking-[0.18em] text-[var(--text-muted)]">
                      Dataset: {registeredModel.dataset_id}
                    </p>
                  </button>
                ))}
              </div>
            </Panel>

            <Panel tone="muted">
              <div className="grid gap-5 lg:grid-cols-[1fr_250px] lg:items-start">
                <div>
                  <p className="eyebrow">CSV prefill</p>
                  <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                    Load a compatible feature row
                  </h2>
                  <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                    This single-sample flow accepts extra columns, but every required feature must be present.
                  </p>
                </div>
                <StatusPill tone="positive">first row prediction</StatusPill>
              </div>

              <label className="mt-7 flex min-h-[190px] cursor-pointer flex-col items-center justify-center rounded-[30px] border border-dashed border-[var(--border-strong)] bg-white/48 px-6 text-center transition hover:bg-white/70">
                <div className="grid h-16 w-16 place-items-center rounded-full bg-[var(--accent-soft)] text-2xl text-[var(--accent-strong)]">
                  <FaUpload />
                </div>
                <p className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
                  {fileName || "Choose a CSV row to preload"}
                </p>
                <p className="mt-3 max-w-md text-sm leading-7 text-[var(--text-muted)]">
                  For full external dataset testing, use the Dataset Test page.
                </p>
                <span className="mt-5 inline-flex items-center gap-2 rounded-full border border-[var(--border-strong)] bg-white/80 px-4 py-2 text-sm font-semibold text-[var(--accent-strong)]">
                  <FaFileCsv className="text-xs" />
                  Browse CSV
                </span>
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </label>
            </Panel>

            <Panel>
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="eyebrow">Voice feature input</p>
                  <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                    Grouped fields first, exact columns when needed
                  </h2>
                  <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                    Group names explain the acoustic family; labels still map to exact deployed feature columns.
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <ActionButton
                    variant={inputMode === "grouped" ? "primary" : "secondary"}
                    onClick={() => setInputMode("grouped")}
                  >
                    <FaLayerGroup className="mr-2 text-xs" />
                    Grouped
                  </ActionButton>
                  <ActionButton
                    variant={inputMode === "advanced" ? "primary" : "secondary"}
                    onClick={() => setInputMode("advanced")}
                  >
                    <FaColumns className="mr-2 text-xs" />
                    Exact columns
                  </ActionButton>
                </div>
              </div>

              <div className="mt-7 flex flex-wrap gap-3">
                <ActionButton variant="secondary" onClick={handleLoadSample}>
                  Load saved sample
                </ActionButton>
                <Link href="/dataset-test" className="button-ghost inline-flex items-center rounded-full px-5 py-3 text-sm font-semibold">
                  Open dataset test
                </Link>
              </div>

              <div className="mt-8 max-h-[680px] space-y-5 overflow-y-auto pr-2">
                {inputMode === "grouped"
                  ? groups.map((group) => (
                      <div key={group.name} className="feature-group">
                        <div className="mb-5 flex items-start gap-3">
                          <div className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-[var(--accent-soft)] text-[var(--accent-strong)]">
                            <FaWaveSquare />
                          </div>
                          <div>
                            <h3 className="text-xl font-semibold text-[var(--text-strong)]">
                              {group.name}
                            </h3>
                            <p className="mt-1 text-sm leading-7 text-[var(--text-muted)]">
                              {group.description}
                            </p>
                          </div>
                        </div>
                        <div className="grid gap-4 md:grid-cols-2">
                          {group.features.map((feature) => (
                            <div key={feature}>
                              <label className="field-label" htmlFor={feature}>
                                {friendlyFeatureName(feature)}
                              </label>
                              <input
                                id={feature}
                                type="number"
                                name={feature}
                                value={featureValues[feature] || ""}
                                onChange={handleInputChange}
                                placeholder="0.0"
                                step="any"
                                className="input-shell"
                              />
                              <p className="mt-2 truncate text-xs text-[var(--text-muted)]">
                                Column: {feature}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))
                  : (
                    <div className="grid gap-4 md:grid-cols-2">
                      {(featuresInfo?.expected_features ?? []).map((feature) => (
                        <div key={feature}>
                          <label className="field-label" htmlFor={feature}>
                            {feature}
                          </label>
                          <input
                            id={feature}
                            type="number"
                            name={feature}
                            value={featureValues[feature] || ""}
                            onChange={handleInputChange}
                            placeholder="0.0"
                            step="any"
                            className="input-shell"
                          />
                        </div>
                      ))}
                    </div>
                  )}
              </div>

              <ActionButton
                onClick={handlePredict}
                disabled={loading || !featuresInfo}
                className="mt-8 w-full"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                    Analyzing feature profile
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <FaPlay className="text-xs" />
                    Run Parkinson&apos;s prediction
                  </span>
                )}
              </ActionButton>
            </Panel>
          </div>

          <div className="space-y-6">
            {displayedPrediction && displayedRisk ? (
              <Panel tone="strong" className="sticky top-28">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <StatusPill tone={riskTone(displayedRisk)}>
                      {displayedRisk}
                    </StatusPill>
                    <h2 className="mt-5 font-display text-4xl text-[#fff8ea]">
                      {predictionLabel(displayedPrediction.prediction)}
                    </h2>
                    <p className="mt-3 text-sm leading-7 text-[rgba(255,248,234,0.74)]">
                      {confidenceLabel(displayedPrediction.confidence)} using {reviewModelName}
                    </p>
                  </div>
                  {uploadedFileName ? (
                    <StatusPill tone="accent">{uploadedFileName}</StatusPill>
                  ) : null}
                </div>

                <div className="mt-8 flex justify-center">
                  <ProbabilityGauge probability={displayedPrediction.probability} />
                </div>

                <div className="mt-8 space-y-3">
                  <div className="data-row !border-[rgba(255,248,234,0.08)] !text-[rgba(255,248,234,0.72)]">
                    <span>Probability</span>
                    <span className="font-semibold text-[#fff8ea]">
                      {(displayedPrediction.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="data-row !border-[rgba(255,248,234,0.08)] !text-[rgba(255,248,234,0.72)]">
                    <span>Confidence</span>
                    <span className="font-semibold text-[#fff8ea]">
                      {(displayedPrediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="data-row !border-[rgba(255,248,234,0.08)] !text-[rgba(255,248,234,0.72)]">
                    <span>Training dataset</span>
                    <span className="font-semibold text-[#fff8ea]">
                      {featuresInfo?.training_dataset ?? "pd_speech_features.csv"}
                    </span>
                  </div>
                </div>

                <div className="mt-8 flex flex-wrap gap-3">
                  <Link
                    href="/prediction"
                    className="inline-flex items-center justify-center rounded-full border border-[rgba(255,248,234,0.16)] bg-[rgba(255,248,234,0.08)] px-4 py-2 text-sm font-semibold text-[#fff8ea]"
                  >
                    View prediction detail
                  </Link>
                  <Link
                    href="/explainability"
                    className="inline-flex items-center justify-center rounded-full border border-[rgba(255,248,234,0.16)] bg-[rgba(255,248,234,0.08)] px-4 py-2 text-sm font-semibold text-[#fff8ea]"
                  >
                    Review SHAP explanation
                  </Link>
                </div>
              </Panel>
            ) : (
              <EmptyState
                icon={<FaFlask />}
                title="No prediction yet"
                description="Choose a registered model, load the sample, or enter grouped voice features to create a live result."
              />
            )}

            <Panel>
              <p className="eyebrow">Demo guardrail</p>
              <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                What this page does not do
              </h2>
              <p className="mt-4 text-sm leading-7 text-[var(--text-muted)]">
                This v1 does not record raw microphone audio. It accepts deployed acoustic
                feature values that already match the model&apos;s selected feature schema.
              </p>
              <div className="mt-6 space-y-3">
                <div className="data-row">
                  <span>Model training source</span>
                  <strong>pd_speech_features.csv</strong>
                </div>
                <div className="data-row">
                  <span>External dataset testing</span>
                  <strong>Dataset Test route</strong>
                </div>
                <div className="data-row">
                  <span>Name changes</span>
                  <strong>backend registry config</strong>
                </div>
              </div>
            </Panel>
          </div>
        </div>
      </SiteContainer>
    </div>
  );
}
