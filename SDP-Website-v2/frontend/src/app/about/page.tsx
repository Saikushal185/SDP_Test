"use client";

import {
  FaArrowDown,
  FaBrain,
  FaChartBar,
  FaCogs,
  FaDatabase,
  FaDesktop,
  FaLightbulb,
  FaMicroscope,
  FaShieldAlt,
} from "react-icons/fa";
import {
  PageIntro,
  Panel,
  SectionHeading,
  SiteContainer,
  StatusPill,
} from "@/components/site/ui";

const pipelineSteps = [
  {
    icon: FaDatabase,
    title: "Speech Dataset",
    description:
      "PD speech feature records establish the input distribution used by the deployed interface.",
    color: "from-[#55737a] to-[#365258]",
  },
  {
    icon: FaChartBar,
    title: "Feature Extraction",
    description:
      "Acoustic descriptors such as PPE, DFA, RPDE, jitter, and shimmer are prepared for inference.",
    color: "from-[#4f6b72] to-[#274149]",
  },
  {
    icon: FaCogs,
    title: "Machine Learning Models",
    description:
      "Saved classifiers such as Logistic Regression, SVM, Random Forest, and XGBoost provide benchmark context.",
    color: "from-[#45636a] to-[#1f3840]",
  },
  {
    icon: FaBrain,
    title: "Prediction",
    description:
      "The interface produces a Parkinson's likelihood score together with confidence and model context.",
    color: "from-[#38555d] to-[#1a3138]",
  },
  {
    icon: FaLightbulb,
    title: "Explainable AI",
    description:
      "Grouped SHAP signals and raw feature-level contributions surface why the score moved.",
    color: "from-[#6d7053] to-[#484f34]",
  },
  {
    icon: FaDesktop,
    title: "Research Dashboard",
    description:
      "The frontend packages prediction, interpretation, and benchmark review into one polished route structure.",
    color: "from-[#2f5058] to-[#182a31]",
  },
];

export default function AboutPage() {
  return (
    <div className="pb-20">
      <PageIntro
        eyebrow="Methodology"
        title="Explainable Parkinson&apos;s detection from speech features."
        description="This route presents the project as a reviewable research system: data preparation, model selection, explainability, and the web interface that ties those stages together."
        meta={[
          { label: "Primary modality", value: "Speech features" },
          { label: "Interpretability", value: "SHAP-based" },
          { label: "Presentation mode", value: "Research showcase" },
        ]}
      />

      <SiteContainer className="space-y-14 pt-10">
        <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <Panel>
            <SectionHeading
              eyebrow="Project overview"
              title="Built to balance predictive performance with traceable reasoning."
              description="The platform combines machine learning inference with a presentation layer designed for scrutiny. Reviewers can move from raw feature input to model output, grouped SHAP drivers, and benchmark results without leaving the interface."
            />
            <div className="mt-8 space-y-4 text-sm leading-8 text-[var(--text-muted)]">
              <p>
                The emphasis is not just on whether the model predicts a likely
                Parkinson&apos;s pattern, but on how that output can be explained
                in ways that are useful for faculty review and technical
                discussion.
              </p>
              <p>
                By pairing plain-language summaries with advanced feature-level
                tables, the system supports both fast scanning and deeper model
                interpretation during demos or project evaluations.
              </p>
            </div>
          </Panel>

          <Panel tone="strong">
            <StatusPill tone="accent">Review posture</StatusPill>
            <h2 className="mt-6 font-display text-4xl text-[#f8f5ef]">
              Trust comes from visibility.
            </h2>
            <div className="mt-8 space-y-4 text-sm leading-7 text-[rgba(248,245,239,0.72)]">
              <div className="data-row !border-[rgba(248,245,239,0.08)]">
                <span>Prediction route</span>
                <span className="font-semibold text-[#f8f5ef]">
                  Outcome + confidence
                </span>
              </div>
              <div className="data-row !border-[rgba(248,245,239,0.08)]">
                <span>Explainability route</span>
                <span className="font-semibold text-[#f8f5ef]">
                  Grouped SHAP evidence
                </span>
              </div>
              <div className="data-row !border-[rgba(248,245,239,0.08)]">
                <span>Performance route</span>
                <span className="font-semibold text-[#f8f5ef]">
                  Readable model comparison
                </span>
              </div>
            </div>
          </Panel>
        </section>

        <section>
          <SectionHeading
            eyebrow="Pipeline"
            title="A six-step pathway from dataset to reviewable dashboard."
            description="The pipeline below shows the research framing behind the interface, not just the UI screens."
          />
          <div className="mx-auto mt-10 max-w-3xl">
            {pipelineSteps.map((step, index) => (
              <div key={step.title}>
                <div className="flex items-start gap-4">
                  <div
                    className={`flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-[22px] bg-gradient-to-br ${step.color} shadow-[0_16px_28px_rgba(17,33,38,0.12)]`}
                  >
                    <step.icon className="text-xl text-white" />
                  </div>
                  <div className="site-panel-muted flex-1">
                    <div className="flex items-center justify-between gap-4">
                      <h3 className="text-lg font-semibold text-[var(--text-strong)]">
                        {step.title}
                      </h3>
                      <span className="text-[0.72rem] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
                        Step {index + 1}
                      </span>
                    </div>
                    <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
                      {step.description}
                    </p>
                  </div>
                </div>
                {index < pipelineSteps.length - 1 && (
                  <div className="ml-7 flex items-center py-3">
                    <div className="h-6 w-px bg-[var(--border-strong)]" />
                    <FaArrowDown className="-ml-[7px] text-sm text-[var(--accent)]" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        <section className="grid gap-5 md:grid-cols-3">
          <Panel>
            <FaDesktop className="text-2xl text-[var(--accent-strong)]" />
            <h3 className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
              Frontend
            </h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
              Next.js, Tailwind CSS, and Recharts provide the responsive shell,
              page hierarchy, and model visualizations.
            </p>
          </Panel>
          <Panel>
            <FaCogs className="text-2xl text-[var(--accent-strong)]" />
            <h3 className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
              Backend
            </h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
              Flask exposes the prediction, explainability, features, and metrics
              endpoints that the frontend depends on.
            </p>
          </Panel>
          <Panel>
            <FaLightbulb className="text-2xl text-[var(--accent-strong)]" />
            <h3 className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
              Interpretability
            </h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
              SHAP values and grouped feature summaries make the model output more
              defensible during technical review.
            </p>
          </Panel>
        </section>

        <section className="grid gap-5 md:grid-cols-3">
          <Panel tone="muted">
            <FaDatabase className="text-2xl text-[var(--accent-strong)]" />
            <h3 className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
              Dataset credibility
            </h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
              Voice-derived measurements anchor the project in a concrete clinical
              signal domain.
            </p>
          </Panel>
          <Panel tone="muted">
            <FaMicroscope className="text-2xl text-[var(--accent-strong)]" />
            <h3 className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
              Reviewer usability
            </h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
              The interface is structured to let judges scan the story quickly and
              inspect the details when needed.
            </p>
          </Panel>
          <Panel tone="muted">
            <FaShieldAlt className="text-2xl text-[var(--accent-strong)]" />
            <h3 className="mt-5 text-lg font-semibold text-[var(--text-strong)]">
              Scope clarity
            </h3>
            <p className="mt-3 text-sm leading-7 text-[var(--text-muted)]">
              This is a research and decision-support demo, not a clinical
              diagnosis tool.
            </p>
          </Panel>
        </section>
      </SiteContainer>
    </div>
  );
}
