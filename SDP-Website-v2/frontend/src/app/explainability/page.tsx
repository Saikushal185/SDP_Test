"use client";

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
import { FaChartBar, FaLightbulb } from "react-icons/fa";
import {
  EmptyState,
  LinkButton,
  PageIntro,
  Panel,
  SiteContainer,
  StatusPill,
} from "@/components/site/ui";
import { usePrediction } from "@/context/PredictionContext";
import {
  buildAdvancedFeatureRows,
  buildDriverSentence,
  buildGroupedShapData,
  confidenceLabel,
  formatModelName,
  formatShapDirection,
  predictionLabel,
  riskLabel,
  riskTone,
} from "@/lib/prediction-utils";

export default function ExplainabilityPage() {
  const { explanation, features, model, modelDisplayName, prediction } = usePrediction();

  if (!prediction || !explanation) {
    return (
      <div className="pb-20">
        <PageIntro
          eyebrow="Explainability"
          title="Interpretation is available once a prediction has been generated."
          description="This route is meant to explain the latest output in plain language first, then expose the raw feature-level SHAP details for technical review."
        />
        <SiteContainer className="pt-10">
          <EmptyState
            icon={<FaLightbulb />}
            title="No explanation available yet"
            description="Run the analysis from the upload route first so the app can show which groups of voice signals influenced the result."
            action={<LinkButton href="/upload">Go to upload</LinkButton>}
          />
        </SiteContainer>
      </div>
    );
  }

  const groupedDrivers = buildGroupedShapData(explanation);
  const advancedRows = buildAdvancedFeatureRows(explanation, features);
  const strongestIncreases = groupedDrivers
    .filter((item) => item.value > 0)
    .slice(0, 3);
  const strongestDecreases = groupedDrivers
    .filter((item) => item.value < 0)
    .slice(0, 3);
  const risk = riskLabel(prediction.probability);
  const activeModelName = modelDisplayName || formatModelName(model || "xgboost");

  return (
    <div className="pb-20">
      <PageIntro
        eyebrow="Explainability"
        title="A plain-language interpretation layer for the latest result."
        description="The goal of this screen is to make the model legible: first through grouped signal families, then through raw feature-level contribution details."
        meta={[
          { label: "Risk band", value: risk },
          { label: "Model", value: activeModelName },
          {
            label: "Confidence",
            value: `${(prediction.confidence * 100).toFixed(1)}%`,
          },
        ]}
      />

      <SiteContainer className="space-y-6 pt-10">
        <div className="grid gap-6 lg:grid-cols-[1.18fr_0.82fr]">
          <Panel>
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[var(--accent-soft)] text-[var(--accent-strong)]">
                <FaChartBar />
              </div>
              <div>
                <p className="eyebrow">Grouped explanation</p>
                <h2 className="mt-2 text-2xl font-semibold text-[var(--text-strong)]">
                  What most influenced the score
                </h2>
                <p className="mt-2 text-sm leading-7 text-[var(--text-muted)]">
                  Red bars pushed the model toward a Parkinson&apos;s-positive
                  outcome. Green bars softened that score.
                </p>
              </div>
            </div>

            <div className="mt-8" style={{ height: groupedDrivers.length * 58 + 24 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={groupedDrivers}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(17,33,38,0.08)" />
                  <XAxis type="number" tick={{ fontSize: 12 }} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    tick={{ fontSize: 11, fontWeight: 600 }}
                    width={115}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: "18px",
                      border: "1px solid rgba(17, 33, 38, 0.08)",
                      boxShadow: "0 20px 40px rgba(17, 33, 38, 0.10)",
                    }}
                    formatter={(value, _, payload) => {
                      const numericValue = Number(
                        Array.isArray(value) ? value[0] : value ?? 0
                      );
                      const point = payload?.payload as
                        | { featureCount?: number }
                        | undefined;
                      return [
                        `${numericValue > 0 ? "+" : ""}${numericValue.toFixed(4)}`,
                        `${point?.featureCount ?? 0} related feature${
                          point?.featureCount === 1 ? "" : "s"
                        }`,
                      ];
                    }}
                  />
                  <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={30}>
                    {groupedDrivers.map((entry) => (
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

          <div className="space-y-6">
            <Panel tone="strong">
              <StatusPill tone={riskTone(risk)}>{risk}</StatusPill>
              <h2 className="mt-5 font-display text-4xl text-[#f8f5ef]">
                {predictionLabel(prediction.prediction)}
              </h2>
              <p className="mt-3 text-sm leading-7 text-[rgba(248,245,239,0.74)]">
                {confidenceLabel(prediction.confidence)} using{" "}
                {activeModelName}
              </p>
              <div className="mt-8 space-y-3">
                <div className="data-row !border-[rgba(248,245,239,0.08)] !text-[rgba(248,245,239,0.72)]">
                  <span>Probability</span>
                  <span className="font-semibold text-[#f8f5ef]">
                    {(prediction.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="data-row !border-[rgba(248,245,239,0.08)] !text-[rgba(248,245,239,0.72)]">
                  <span>Confidence</span>
                  <span className="font-semibold text-[#f8f5ef]">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </Panel>

            <Panel>
              <h2 className="text-xl font-semibold text-[var(--text-strong)]">
                Main reasons the score increased
              </h2>
              <div className="mt-5 space-y-3">
                {strongestIncreases.map((item) => (
                  <div
                    key={item.name}
                    className="rounded-[22px] border border-[rgba(142,75,67,0.16)] bg-[rgba(142,75,67,0.1)] px-4 py-4"
                  >
                    <p className="font-semibold text-[var(--danger)]">{item.name}</p>
                    <p className="mt-2 text-sm leading-7 text-[rgba(142,75,67,0.92)]">
                      {buildDriverSentence(item)}
                    </p>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel tone="muted">
              <h2 className="text-xl font-semibold text-[var(--text-strong)]">
                Main reasons the score decreased
              </h2>
              <div className="mt-5 space-y-3">
                {strongestDecreases.map((item) => (
                  <div
                    key={item.name}
                    className="rounded-[22px] border border-[rgba(47,106,85,0.16)] bg-[rgba(47,106,85,0.1)] px-4 py-4"
                  >
                    <p className="font-semibold text-[var(--success)]">{item.name}</p>
                    <p className="mt-2 text-sm leading-7 text-[rgba(47,106,85,0.92)]">
                      {buildDriverSentence(item)}
                    </p>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
        </div>

        <Panel>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="eyebrow">Advanced detail</p>
              <h2 className="mt-3 text-2xl font-semibold text-[var(--text-strong)]">
                Raw feature-level SHAP contributions
              </h2>
              <p className="mt-3 max-w-3xl text-sm leading-7 text-[var(--text-muted)]">
                This table preserves the technical layer beneath the grouped
                explanation so reviewers can connect individual features to the
                summarized story.
              </p>
            </div>
            <LinkButton href="/prediction" variant="secondary">
              Back to prediction summary
            </LinkButton>
          </div>

          <div className="table-shell mt-8 overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th className="text-left">Raw feature</th>
                  <th className="text-left">Friendly group</th>
                  <th className="text-center">Sample value</th>
                  <th className="text-center">SHAP contribution</th>
                  <th className="text-left">Direction</th>
                </tr>
              </thead>
              <tbody>
                {advancedRows.map((row) => (
                  <tr key={row.featureName}>
                    <td className="font-medium">{row.featureName}</td>
                    <td>{row.displayGroup}</td>
                    <td className="text-center">
                      {row.featureValue === null ? "N/A" : row.featureValue.toFixed(4)}
                    </td>
                    <td
                      className={`text-center font-semibold ${
                        row.shapValue >= 0
                          ? "text-[var(--danger)]"
                          : "text-[var(--success)]"
                      }`}
                    >
                      {row.shapValue >= 0 ? "+" : ""}
                      {row.shapValue.toFixed(4)}
                    </td>
                    <td>{formatShapDirection(row.shapValue)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <div className="callout-warning">
          These explanations describe what influenced the model&apos;s score for
          this sample. They do not identify a medical cause or replace a
          clinical diagnosis.
        </div>
      </SiteContainer>
    </div>
  );
}
