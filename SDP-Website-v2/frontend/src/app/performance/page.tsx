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
import { FaChartBar, FaShieldAlt, FaTrophy } from "react-icons/fa";
import { EmptyState, PageIntro, Panel, SiteContainer } from "@/components/site/ui";
import {
  fetchCrossDatasetSummary,
  fetchMetrics,
  fetchModelRegistry,
  type CrossDatasetSummary,
  type MetricsData,
  type ModelRegistryData,
} from "@/lib/api";
import {
  buildPlainLanguageMetricCards,
  getMetricCopy,
  modelDisplayName,
} from "@/lib/prediction-utils";

const metricKeys = ["accuracy", "recall", "precision"] as const;
const chartColors = ["#1d4850", "#55737a", "#8d6234"];

export default function PerformancePage() {
  const [data, setData] = useState<MetricsData | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryData | null>(null);
  const [crossSummary, setCrossSummary] = useState<CrossDatasetSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    Promise.all([fetchMetrics(), fetchModelRegistry(), fetchCrossDatasetSummary()])
      .then(([metricsData, registryData, summaryData]) => {
        if (cancelled) {
          return;
        }
        setData(metricsData);
        setRegistry(registryData);
        setCrossSummary(summaryData);
      })
      .catch((fetchError) => {
        if (!cancelled) {
          setError(
            fetchError instanceof Error
              ? fetchError.message
              : "Failed to load performance metrics."
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="pb-20">
        <PageIntro
          eyebrow="Performance review"
          title="Saved benchmark metrics are currently unavailable."
          description="The page can still render once the backend metrics endpoint is reachable."
        />
        <SiteContainer className="pt-10">
          <EmptyState
            icon={<FaChartBar />}
            title="Unable to load performance metrics"
            description={error}
          />
        </SiteContainer>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="pb-20">
        <PageIntro
          eyebrow="Performance review"
          title="Loading saved model benchmarks."
          description="This route turns the evaluation metrics into a cleaner story for non-technical reviewers."
        />
        <SiteContainer className="pt-10">
          <Panel className="text-center text-sm text-[var(--text-muted)]">
            Loading performance data...
          </Panel>
        </SiteContainer>
      </div>
    );
  }

  const modelKeys = Object.keys(data.models);
  const bestModelKey = data.best_model;
  const bestModelName = modelDisplayName(bestModelKey, registry?.models ?? []);
  const bestModelMetrics = data.models[bestModelKey];
  const bestModelCards = bestModelMetrics
    ? buildPlainLanguageMetricCards(bestModelMetrics)
    : [];

  const metricCopy = getMetricCopy();
  const comparisonData = metricKeys.map((metricKey) => {
    const row: Record<string, string | number> = {
      label: metricCopy[metricKey].label,
    };

    modelKeys.forEach((modelKey) => {
      row[modelKey] = +(data.models[modelKey][metricKey] * 100).toFixed(1);
    });

    return row;
  });

  return (
    <div className="pb-20">
      <PageIntro
        eyebrow="Performance review"
        title="Benchmark evidence for the one-dataset model registry."
        description="This view compares only registered models trained on pd_speech_features.csv, then shows external artifact schemas as compatibility context."
        meta={[
          { label: "Best model", value: bestModelName },
          {
            label: "Training dataset",
            value: registry?.training_dataset ?? "pd_speech_features.csv",
          },
          { label: "Compared models", value: `${modelKeys.length}` },
        ]}
      />

      <SiteContainer className="space-y-6 pt-10">
        <Panel tone="strong">
          <div className="grid gap-8 lg:grid-cols-[0.95fr_1.05fr]">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[rgba(248,245,239,0.08)] text-[#f8f5ef]">
                  <FaTrophy />
                </div>
                <div>
                  <p className="text-[0.72rem] uppercase tracking-[0.2em] text-[rgba(248,245,239,0.44)]">
                    Best model for this study
                  </p>
                  <h2 className="mt-2 font-display text-4xl text-[#f8f5ef]">
                    {bestModelName}
                  </h2>
                </div>
              </div>
              <p className="mt-6 max-w-xl text-sm leading-7 text-[rgba(248,245,239,0.74)]">
                This model performed best in the saved evaluation output for
                pd_speech_features.csv. Display names are read from the backend
                registry, not hardcoded in the page.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              {bestModelCards.map((card, index) => (
                <div
                  key={card.key}
                  className="rounded-[24px] border border-[rgba(248,245,239,0.1)] bg-[rgba(248,245,239,0.06)] p-5"
                >
                  <div
                    className="flex h-11 w-11 items-center justify-center rounded-full"
                    style={{
                      backgroundColor: `${chartColors[index]}24`,
                      color: "#f8f5ef",
                    }}
                  >
                    <FaShieldAlt />
                  </div>
                  <p className="mt-5 text-sm font-medium text-[rgba(248,245,239,0.68)]">
                    {card.label}
                  </p>
                  <p className="mt-3 font-display text-4xl text-[#f8f5ef]">
                    {(card.value * 100).toFixed(1)}%
                  </p>
                  <p className="mt-3 text-sm leading-7 text-[rgba(248,245,239,0.68)]">
                    {card.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        <Panel>
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[var(--accent-soft)] text-[var(--accent-strong)]">
              <FaChartBar />
            </div>
            <div>
              <p className="eyebrow">Readable comparison</p>
              <h2 className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">
                Plain-language metrics first
              </h2>
              <p className="mt-2 text-sm leading-7 text-[var(--text-muted)]">
                These three measures give reviewers the quickest way to compare
                the saved models without reading a dense technical table first.
              </p>
            </div>
          </div>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {metricKeys.map((metricKey) => (
              <div
                key={metricKey}
                className="rounded-[24px] border border-[var(--border-subtle)] bg-white/60 p-5"
              >
                <p className="text-lg font-semibold text-[var(--text-strong)]">
                  {metricCopy[metricKey].label}
                </p>
                <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                  {metricCopy[metricKey].description}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-8 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparisonData}
                margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(17,33,38,0.08)" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} interval={0} />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                  label={{
                    value: "Score (%)",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 12,
                  }}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: "18px",
                    border: "1px solid rgba(17, 33, 38, 0.08)",
                    boxShadow: "0 20px 40px rgba(17, 33, 38, 0.10)",
                  }}
                  formatter={(value, name) => [
                    `${Number(value).toFixed(1)}%`,
                    modelDisplayName(String(name), registry?.models ?? []),
                  ]}
                />
                {modelKeys.map((modelKey, index) => (
                  <Bar
                    key={modelKey}
                    dataKey={modelKey}
                    name={modelDisplayName(modelKey, registry?.models ?? [])}
                    radius={[8, 8, 0, 0]}
                    fill={chartColors[index % chartColors.length]}
                  >
                    {comparisonData.map((entry) => (
                      <Cell
                        key={`${modelKey}-${String(entry.label)}`}
                        fill={chartColors[index % chartColors.length]}
                      />
                    ))}
                  </Bar>
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel tone="muted">
          <p className="eyebrow">Advanced detail</p>
          <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
            Original technical evaluation metrics
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-[var(--text-muted)]">
            The table below preserves the saved benchmark data in its more
            technical form for direct model comparison.
          </p>

          <div className="table-shell mt-8 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th className="text-left">Model</th>
                  <th className="text-center">Accuracy</th>
                  <th className="text-center">Precision</th>
                  <th className="text-center">Recall</th>
                  <th className="text-center">F1 Score</th>
                  <th className="text-center">AUC</th>
                  <th className="text-left">Training Dataset</th>
                </tr>
              </thead>
              <tbody>
                {modelKeys.map((modelKey) => {
                  const metrics = data.models[modelKey];
                  const isBest = modelKey === bestModelKey;
                  return (
                    <tr
                      key={modelKey}
                      className={isBest ? "bg-[rgba(29,72,80,0.05)]" : undefined}
                    >
                      <td>
                        <div className="flex items-center gap-3">
                          <span className="font-medium">
                            {modelDisplayName(modelKey, registry?.models ?? [])}
                          </span>
                          {isBest ? (
                            <span className="rounded-full border border-[rgba(29,72,80,0.16)] bg-[rgba(29,72,80,0.08)] px-2.5 py-1 text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-[var(--accent-strong)]">
                              Best
                            </span>
                          ) : null}
                        </div>
                      </td>
                      <td className="text-center">
                        {(metrics.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="text-center">
                        {(metrics.precision * 100).toFixed(1)}%
                      </td>
                      <td className="text-center">
                        {(metrics.recall * 100).toFixed(1)}%
                      </td>
                      <td className="text-center">
                        {(metrics.f1 * 100).toFixed(1)}%
                      </td>
                      <td className="text-center">{metrics.auc.toFixed(3)}</td>
                      <td>pd_speech_features.csv</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel>
          <p className="eyebrow">Cross-dataset context</p>
          <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
            Compatibility is reported separately from model performance
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-[var(--text-muted)]">
            Existing local artifacts may have strong within-dataset results, but strict
            direct testing only applies when their schema contains every deployed
            required feature.
          </p>

          <div className="table-shell mt-8 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th className="text-left">Dataset</th>
                  <th className="text-center">Features</th>
                  <th className="text-center">Required overlap</th>
                  <th className="text-center">Missing required</th>
                  <th className="text-center">Strict status</th>
                </tr>
              </thead>
              <tbody>
                {(crossSummary?.datasets ?? []).map((dataset) => (
                  <tr key={dataset.dataset_id}>
                    <td>{dataset.dataset_id.replaceAll("_", " ")}</td>
                    <td className="text-center">{dataset.feature_count}</td>
                    <td className="text-center">{dataset.required_overlap}</td>
                    <td className="text-center">{dataset.missing_required_count}</td>
                    <td className="text-center">
                      {dataset.strict_compatible ? "Compatible" : "Blocked"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </SiteContainer>
    </div>
  );
}
