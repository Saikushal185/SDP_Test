"use client";

import { useEffect, useState } from "react";
import {
  FaArrowRight,
  FaDatabase,
  FaFileCsv,
  FaFingerprint,
  FaMicroscope,
  FaWaveSquare,
} from "react-icons/fa";
import {
  fetchCrossDatasetSummary,
  fetchModelInfo,
  fetchModelRegistry,
  type CrossDatasetSummary,
  type ModelInfo,
  type ModelRegistryData,
} from "@/lib/api";
import { LinkButton, Panel, SectionHeading, SiteContainer, StatusPill } from "@/components/site/ui";
import { modelDisplayName } from "@/lib/prediction-utils";

const workflow = [
  {
    title: "Choose a registered model",
    description: "Display names come from backend config, but inference still uses stable model keys.",
  },
  {
    title: "Enter friendly voice features",
    description: "Grouped acoustic fields keep the demo readable while preserving exact deployed columns.",
  },
  {
    title: "Test compatible CSVs only",
    description: "External datasets must contain every required feature from the training dataset.",
  },
];

export default function Home() {
  const [registry, setRegistry] = useState<ModelRegistryData | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [crossSummary, setCrossSummary] = useState<CrossDatasetSummary | null>(null);

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      fetchModelRegistry(),
      fetchModelInfo(),
      fetchCrossDatasetSummary(),
    ])
      .then(([registryData, infoData, summaryData]) => {
        if (cancelled) {
          return;
        }
        setRegistry(registryData);
        setInfo(infoData);
        setCrossSummary(summaryData);
      })
      .catch(console.error);

    return () => {
      cancelled = true;
    };
  }, []);

  const models = registry?.models ?? [];
  const bestModel = registry?.best_model ?? info?.best_model ?? "xgboost";
  const strictCompatibleCount =
    crossSummary?.datasets.filter((dataset) => dataset.strict_compatible).length ?? 0;

  return (
    <div className="pb-20">
      <section className="full-bleed-hero">
        <div className="hero-field" />
        <SiteContainer className="relative grid min-h-[calc(100svh-72px)] gap-12 py-14 lg:grid-cols-[1.02fr_0.98fr] lg:items-center lg:py-20">
          <div className="hero-reveal max-w-4xl">
            <StatusPill tone="accent" className="mb-7">
              <FaDatabase className="text-xs" />
              trained only on pd_speech_features.csv
            </StatusPill>
            <h1 className="display-title">
              Parkinson&apos;s voice models with honest dataset boundaries.
            </h1>
            <p className="mt-8 max-w-2xl text-lg leading-8 text-[rgba(255,248,234,0.78)] sm:text-xl">
              V2 turns the project into a review-ready lab: rename model labels from config,
              enter grouped acoustic features, and test external CSVs only when their schema
              truly matches the training dataset.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <LinkButton href="/upload" className="!bg-[#fff8ea] !text-[var(--accent-strong)]">
                Start analysis
                <FaArrowRight className="ml-2 text-xs" />
              </LinkButton>
              <LinkButton
                href="/dataset-test"
                variant="ghost"
                className="!border-[rgba(255,248,234,0.2)] !text-[#fff8ea]"
              >
                Test another CSV
              </LinkButton>
            </div>
          </div>

          <div className="hero-reveal [animation-delay:120ms]">
            <div className="signal-console">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[0.72rem] font-bold uppercase tracking-[0.22em] text-[rgba(255,248,234,0.45)]">
                    Active registry
                  </p>
                  <h2 className="mt-3 font-display text-4xl text-[#fff8ea]">
                    {modelDisplayName(bestModel, models)}
                  </h2>
                </div>
                <StatusPill tone="positive">configurable names</StatusPill>
              </div>

              <div className="wave-stack mt-10" aria-hidden="true">
                {Array.from({ length: 18 }).map((_, index) => (
                  <span
                    key={index}
                    style={{
                      height: `${22 + ((index * 19) % 76)}px`,
                      animationDelay: `${index * 70}ms`,
                    }}
                  />
                ))}
              </div>

              <div className="mt-10 grid gap-3 sm:grid-cols-3">
                <div className="console-stat">
                  <span>Models</span>
                  <strong>{models.length || "..."}</strong>
                </div>
                <div className="console-stat">
                  <span>Features</span>
                  <strong>{info?.n_selected_features ?? "..."}</strong>
                </div>
                <div className="console-stat">
                  <span>Strict schemas</span>
                  <strong>{crossSummary ? strictCompatibleCount : "..."}</strong>
                </div>
              </div>

              <div className="mt-8 space-y-4">
                {workflow.map((step, index) => (
                  <div key={step.title} className="flex gap-4">
                    <div className="mt-1 grid h-7 w-7 shrink-0 place-items-center rounded-full border border-[rgba(255,248,234,0.16)] bg-[rgba(255,248,234,0.08)] text-xs font-bold text-[#fff8ea]">
                      {index + 1}
                    </div>
                    <div className="border-b border-[rgba(255,248,234,0.08)] pb-4 last:border-b-0 last:pb-0">
                      <p className="font-semibold text-[#fff8ea]">{step.title}</p>
                      <p className="mt-1 text-sm leading-7 text-[rgba(255,248,234,0.66)]">
                        {step.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </SiteContainer>
      </section>

      <SiteContainer className="space-y-20 pt-16">
        <section>
          <SectionHeading
            eyebrow="V2 contract"
            title="The interface now separates model labels, model artifacts, and dataset compatibility."
            description="That distinction matters for a research demo: viewers can change presentation names without accidentally implying that models were trained on multiple datasets."
          />
          <div className="mt-10 grid gap-5 lg:grid-cols-3">
            <Panel>
              <FaFingerprint className="text-3xl text-[var(--accent-strong)]" />
              <h3 className="mt-5 text-2xl font-semibold text-[var(--text-strong)]">
                Config model names
              </h3>
              <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                Edit `backend/model_registry.json` to change display labels while keeping stable inference keys.
              </p>
            </Panel>
            <Panel tone="muted">
              <FaFileCsv className="text-3xl text-[var(--accent-strong)]" />
              <h3 className="mt-5 text-2xl font-semibold text-[var(--text-strong)]">
                Strict dataset tests
              </h3>
              <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                External CSVs are blocked until every deployed training feature is present and numeric.
              </p>
            </Panel>
            <Panel>
              <FaWaveSquare className="text-3xl text-[var(--accent-strong)]" />
              <h3 className="mt-5 text-2xl font-semibold text-[var(--text-strong)]">
                Voice-friendly fields
              </h3>
              <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                Feature groups explain the acoustic meaning before users open the exact-column mode.
              </p>
            </Panel>
          </div>
        </section>

        <section className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr] lg:items-start">
          <div>
            <SectionHeading
              eyebrow="Registered models"
              title="Only models trained on the selected PD speech dataset appear in the app."
              description="The registry carries the display label, dataset identity, artifact path, and saved metrics that the frontend renders."
            />
            <div className="mt-7 flex flex-wrap gap-3">
              <LinkButton href="/performance" variant="secondary">
                Review performance
              </LinkButton>
              <LinkButton href="/dataset-test" variant="ghost">
                Check CSV compatibility
              </LinkButton>
            </div>
          </div>

          <Panel tone="strong">
            <div className="space-y-4">
              {models.map((model) => (
                <div
                  key={model.key}
                  className="rounded-[26px] border border-[rgba(255,248,234,0.1)] bg-[rgba(255,248,234,0.06)] p-5"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h3 className="text-xl font-semibold text-[#fff8ea]">
                        {model.display_name}
                      </h3>
                      <p className="mt-2 text-sm leading-7 text-[rgba(255,248,234,0.66)]">
                        {model.description || "Registered inference model."}
                      </p>
                    </div>
                    {model.key === bestModel ? <StatusPill tone="accent">best saved</StatusPill> : null}
                  </div>
                  <div className="mt-5 grid gap-3 sm:grid-cols-3">
                    <div className="console-stat">
                      <span>Dataset</span>
                      <strong className="!text-base">{model.dataset_id}</strong>
                    </div>
                    <div className="console-stat">
                      <span>Accuracy</span>
                      <strong>
                        {model.metrics?.accuracy
                          ? `${(model.metrics.accuracy * 100).toFixed(1)}%`
                          : "N/A"}
                      </strong>
                    </div>
                    <div className="console-stat">
                      <span>Feature schema</span>
                      <strong>{model.feature_schema.length}</strong>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </section>

        <section>
          <SectionHeading
            eyebrow="Compatibility context"
            title="Existing external artifacts are visible, but schema mismatch is not hidden."
            description="The summary below explains why strict direct testing blocks datasets that do not share the deployed feature columns."
          />
          <div className="mt-8 grid gap-4 md:grid-cols-2">
            {(crossSummary?.datasets ?? []).slice(0, 4).map((dataset) => (
              <Panel key={dataset.dataset_id} tone={dataset.strict_compatible ? "default" : "muted"}>
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-xl font-semibold text-[var(--text-strong)]">
                      {dataset.dataset_id.replaceAll("_", " ")}
                    </h3>
                    <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                      {dataset.note}
                    </p>
                  </div>
                  <StatusPill tone={dataset.strict_compatible ? "positive" : "caution"}>
                    {dataset.strict_compatible ? "compatible" : "blocked"}
                  </StatusPill>
                </div>
                <div className="mt-5 grid grid-cols-3 gap-3 text-sm">
                  <div className="data-chip">
                    <span>Features</span>
                    <strong>{dataset.feature_count}</strong>
                  </div>
                  <div className="data-chip">
                    <span>Overlap</span>
                    <strong>{dataset.required_overlap}</strong>
                  </div>
                  <div className="data-chip">
                    <span>Missing</span>
                    <strong>{dataset.missing_required_count}</strong>
                  </div>
                </div>
              </Panel>
            ))}
          </div>
        </section>

        <section className="final-band">
          <div>
            <p className="eyebrow !text-[rgba(255,248,234,0.64)]">Ready workflow</p>
            <h2 className="mt-3 font-display text-4xl leading-none text-[#fff8ea] sm:text-5xl">
              Start with the sample, then prove whether another CSV is compatible.
            </h2>
          </div>
          <div className="flex flex-wrap gap-3">
            <LinkButton href="/upload" className="!bg-[#fff8ea] !text-[var(--accent-strong)]">
              Run analysis
            </LinkButton>
            <LinkButton
              href="/dataset-test"
              variant="ghost"
              className="!border-[rgba(255,248,234,0.22)] !text-[#fff8ea]"
            >
              Dataset test
            </LinkButton>
          </div>
        </section>

        <div className="callout-warning flex items-start gap-3">
          <FaMicroscope className="mt-1 shrink-0" />
          <span>
            This research interface reports machine-learning scores from speech features only. It is
            not a medical diagnosis tool.
          </span>
        </div>
      </SiteContainer>
    </div>
  );
}
