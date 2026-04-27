"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FaChartBar, FaCheckCircle, FaWaveSquare } from "react-icons/fa";
import {
  EmptyState,
  LinkButton,
  PageIntro,
  Panel,
  SiteContainer,
  StatusPill,
} from "@/components/site/ui";
import { usePrediction } from "@/context/PredictionContext";
import { fetchMetrics, type MetricsData } from "@/lib/api";
import {
  buildShapChartData,
  confidenceLabel,
  formatModelName,
  predictionLabel,
  riskLabel,
  riskTone,
} from "@/lib/prediction-utils";

export default function PredictionPage() {
  const {
    explanation,
    features,
    model,
    modelDisplayName,
    prediction,
    trainingDataset,
  } = usePrediction();
  const [metrics, setMetrics] = useState<MetricsData | null>(null);

  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(console.error);
  }, []);

  if (!prediction) {
    return (
      <div className="pb-20">
        <PageIntro
          eyebrow="Prediction review"
          title="Prediction detail becomes available after running the analysis."
          description="This route is designed for the second step of the demo: reviewing the model decision, its confidence, and the highest-impact SHAP signals."
        />
        <SiteContainer className="pt-10">
          <EmptyState
            icon={<FaChartBar />}
            title="No prediction available"
          description="Go to the upload route to create a live result first, then return here for the detailed summary."
            action={<LinkButton href="/upload">Go to upload</LinkButton>}
          />
        </SiteContainer>
      </div>
    );
  }

  const shapEntries = buildShapChartData(explanation).slice(0, 15);
  const inputFeatures = Object.entries(features || {}).slice(0, 10);
  const selectedModel = model || metrics?.best_model || "xgboost";
  const selectedModelName = modelDisplayName || formatModelName(selectedModel);
  const modelMetrics = metrics?.models?.[selectedModel];
  const percentage = Math.round(prediction.probability * 100);
  const risk = riskLabel(prediction.probability);

  return (
    <div className="pb-20">
      <PageIntro
        eyebrow="Prediction review"
        title="A clearer summary of the latest model result."
        description="This view foregrounds the decision, confidence, and active model before moving into the supporting metrics and SHAP contributions."
        meta={[
          { label: "Risk band", value: risk },
          { label: "Model", value: selectedModelName },
          {
            label: "Confidence",
            value: `${(prediction.confidence * 100).toFixed(1)}%`,
          },
        ]}
      />

      <SiteContainer className="space-y-6 pt-10">
        <Panel tone="strong" className="overflow-hidden">
          <div className="grid gap-8 lg:grid-cols-[1fr_280px] lg:items-center">
            <div>
              <StatusPill tone={riskTone(risk)}>
                <FaCheckCircle className="text-xs" />
                {risk}
              </StatusPill>
              <h2 className="mt-6 font-display text-5xl text-[#f8f5ef]">
                {predictionLabel(prediction.prediction)}
              </h2>
              <p className="mt-3 max-w-xl text-sm leading-7 text-[rgba(248,245,239,0.74)]">
                {confidenceLabel(prediction.confidence)} with{" "}
                {selectedModelName} as the active model for this
                review session.
              </p>

              <div className="mt-8 grid gap-4 sm:grid-cols-3">
                <div className="rounded-[24px] border border-[rgba(248,245,239,0.1)] bg-[rgba(248,245,239,0.06)] p-5">
                  <p className="text-[0.72rem] uppercase tracking-[0.18em] text-[rgba(248,245,239,0.44)]">
                    Probability
                  </p>
                  <p className="mt-3 font-display text-4xl text-[#f8f5ef]">
                    {percentage}%
                  </p>
                </div>
                <div className="rounded-[24px] border border-[rgba(248,245,239,0.1)] bg-[rgba(248,245,239,0.06)] p-5">
                  <p className="text-[0.72rem] uppercase tracking-[0.18em] text-[rgba(248,245,239,0.44)]">
                    Confidence
                  </p>
                  <p className="mt-3 font-display text-4xl text-[#f8f5ef]">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-[24px] border border-[rgba(248,245,239,0.1)] bg-[rgba(248,245,239,0.06)] p-5">
                  <p className="text-[0.72rem] uppercase tracking-[0.18em] text-[rgba(248,245,239,0.44)]">
                    Model
                  </p>
                  <p className="mt-3 text-2xl font-semibold text-[#f8f5ef]">
                    {selectedModelName}
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-[28px] border border-[rgba(248,245,239,0.1)] bg-[rgba(248,245,239,0.06)] p-6 text-center">
              <p className="text-[0.72rem] uppercase tracking-[0.2em] text-[rgba(248,245,239,0.44)]">
                Confidence score
              </p>
              <p className="mt-4 font-display text-6xl text-[#f8f5ef]">
                {(prediction.confidence * 100).toFixed(0)}
              </p>
              <p className="mt-2 text-sm text-[rgba(248,245,239,0.66)]">
                calibrated review confidence
              </p>
              <div className="mt-8 h-3 rounded-full bg-[rgba(248,245,239,0.12)]">
                <div
                  className="h-full rounded-full bg-[linear-gradient(90deg,rgba(248,245,239,0.62),#f8f5ef)]"
                  style={{ width: `${prediction.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        </Panel>

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel>
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[var(--accent-soft)] text-[var(--accent-strong)]">
                <FaWaveSquare />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-[var(--text-strong)]">
                  Active model context
                </h2>
                <p className="text-sm leading-7 text-[var(--text-muted)]">
                  The chosen model remains visible alongside its saved benchmark
                  metrics and training dataset.
                </p>
              </div>
            </div>

            <div className="mt-8 space-y-3">
              <div className="data-row">
                <span className="text-sm text-[var(--text-muted)]">Model</span>
                <span className="font-semibold text-[var(--text-strong)]">
                  {selectedModelName}
                </span>
              </div>
              <div className="data-row">
                <span className="text-sm text-[var(--text-muted)]">
                  Training dataset
                </span>
                <span className="font-semibold text-[var(--text-strong)]">
                  {trainingDataset || "pd_speech_features.csv"}
                </span>
              </div>
              <div className="data-row">
                <span className="text-sm text-[var(--text-muted)]">Accuracy</span>
                <span className="font-semibold text-[var(--text-strong)]">
                  {modelMetrics
                    ? `${(modelMetrics.accuracy * 100).toFixed(1)}%`
                    : "Loading"}
                </span>
              </div>
              <div className="data-row">
                <span className="text-sm text-[var(--text-muted)]">Recall</span>
                <span className="font-semibold text-[var(--text-strong)]">
                  {modelMetrics
                    ? `${(modelMetrics.recall * 100).toFixed(1)}%`
                    : "Loading"}
                </span>
              </div>
              <div className="data-row">
                <span className="text-sm text-[var(--text-muted)]">F1 score</span>
                <span className="font-semibold text-[var(--text-strong)]">
                  {modelMetrics ? `${(modelMetrics.f1 * 100).toFixed(1)}%` : "Loading"}
                </span>
              </div>
            </div>
          </Panel>

          <Panel tone="muted">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[var(--accent-soft)] text-[var(--accent-strong)]">
                <FaChartBar />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-[var(--text-strong)]">
                  Input features in view
                </h2>
                <p className="text-sm leading-7 text-[var(--text-muted)]">
                  A quick reference of the most visible feature values used for
                  this run.
                </p>
              </div>
            </div>

            <div className="mt-8 space-y-3">
              {inputFeatures.map(([name, value]) => (
                <div key={name} className="data-row">
                  <span className="text-sm text-[var(--text-muted)]">{name}</span>
                  <span className="rounded-full border border-[var(--border-subtle)] bg-white/80 px-3 py-1 text-sm font-semibold text-[var(--text-strong)]">
                    {value.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        {shapEntries.length > 0 ? (
          <Panel>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="eyebrow">Supportive evidence</p>
                <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                  Highest-impact SHAP contributions
                </h2>
                <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--text-muted)]">
                  Positive values increased the Parkinson&apos;s score. Negative
                  values reduced it.
                </p>
              </div>
              <LinkButton href="/explainability" variant="secondary">
                Open explainability view
              </LinkButton>
            </div>

            <div className="mt-8" style={{ height: shapEntries.length * 32 + 40 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={shapEntries}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(17,33,38,0.08)" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    tick={{ fontSize: 10, fontWeight: 600 }}
                    width={130}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: "18px",
                      border: "1px solid rgba(17, 33, 38, 0.08)",
                      boxShadow: "0 20px 40px rgba(17, 33, 38, 0.10)",
                    }}
                    formatter={(value) => {
                      const numericValue = Number(
                        Array.isArray(value) ? value[0] : value ?? 0
                      );
                      return [
                        `${numericValue > 0 ? "+" : ""}${numericValue.toFixed(4)}`,
                        "SHAP",
                      ];
                    }}
                  />
                  <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={20}>
                    {shapEntries.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={entry.value >= 0 ? "#8e4b43" : "#2f6a55"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        ) : null}
      </SiteContainer>
    </div>
  );
}
