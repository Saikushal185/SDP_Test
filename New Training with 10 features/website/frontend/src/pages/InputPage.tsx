import type { Dispatch, SetStateAction } from "react";
import { CheckCircle2, Loader2, Play, RotateCcw } from "lucide-react";
import type { ModelInfo, SamplePayload } from "../types";

type SampleRow = SamplePayload["rows"][number];

type InputPageProps = {
  sampleIndex: number;
  setSampleIndex: (index: number) => void;
  samples: SamplePayload | null;
  editableFeatureNames: string[];
  currentSample: SampleRow | null;
  edits: Record<string, number | null>;
  setEdits: Dispatch<SetStateAction<Record<string, number | null>>>;
  isPredicting: boolean;
  selectedModel: ModelInfo | null;
  onPredict: () => void;
};

type FeatureInfo = {
  label: string;
  unit: string;
  meaning: string;
};

const FEATURE_INFO: Record<string, FeatureInfo> = {
  tqwt_entropy_log_dec_35: {
    label: "Fine voice irregularity",
    unit: "log entropy score",
    meaning: "Shows how uneven the very small vibration patterns are in the voice signal.",
  },
  std_delta_delta_log_energy: {
    label: "Sudden loudness-change variation",
    unit: "standard deviation",
    meaning: "Measures how much the acceleration of voice energy changes across the recording.",
  },
  std_8th_delta_delta: {
    label: "Rapid tone-change variation",
    unit: "standard deviation",
    meaning: "Captures fast changes in one tone/resonance pattern of the voice.",
  },
  mean_MFCC_2nd_coef: {
    label: "Average vocal tone pattern",
    unit: "MFCC average",
    meaning: "Summarizes one of the main sound-shape patterns related to vocal tract resonance.",
  },
  tqwt_TKEO_mean_dec_16: {
    label: "Average rapid voice energy",
    unit: "TKEO mean",
    meaning: "Represents the typical strength of quick energy bursts in a mid-detail signal band.",
  },
  tqwt_entropy_shannon_dec_35: {
    label: "Fine voice disorder",
    unit: "Shannon entropy",
    meaning: "Measures how spread out or unpredictable tiny voice vibrations are.",
  },
  tqwt_TKEO_std_dec_12: {
    label: "Voice energy burst consistency",
    unit: "TKEO standard deviation",
    meaning: "Shows how much quick energy bursts vary in one signal band.",
  },
  tqwt_maxValue_dec_12: {
    label: "Strongest voice vibration peak",
    unit: "maximum band value",
    meaning: "The highest signal peak found in one wave-pattern band.",
  },
  tqwt_entropy_log_dec_11: {
    label: "Lower-band voice irregularity",
    unit: "log entropy score",
    meaning: "Shows unevenness in a broader voice wave-pattern band.",
  },
  tqwt_TKEO_mean_dec_12: {
    label: "Average voice energy bursts",
    unit: "TKEO mean",
    meaning: "Represents the typical rapid energy level in one wave-pattern band.",
  },
};

export function InputPage({
  sampleIndex,
  setSampleIndex,
  samples,
  editableFeatureNames,
  currentSample,
  edits,
  setEdits,
  isPredicting,
  selectedModel,
  onPredict,
}: InputPageProps) {
  const featureNames = editableFeatureNames.slice(0, 10);
  const missingCount = featureNames.filter((feature) => !hasNumericFeatureValue(feature, edits, currentSample)).length;
  const canPredict = Boolean(selectedModel) && featureNames.length === 10 && missingCount === 0;

  function resetToSample() {
    setEdits({});
  }

  function clearInputs() {
    setEdits(Object.fromEntries(featureNames.map((feature) => [feature, null])));
  }

  return (
    <div className="page-stack input-page">
      <header className="page-heading">
        <div>
          <h1>Enter The 10 Speech Feature Values</h1>
          <p>
            These are the exact reduced features used by the trained models. The friendly names explain what each
            number represents, while the technical column names keep the prediction aligned with the dataset.
          </p>
        </div>
        <div className="page-note">
          <CheckCircle2 size={17} />
          <span>Manual single-case prediction, no CSV or raw audio required.</span>
        </div>
      </header>

      <section className="panel feature-form-panel">
        <div className="feature-form-toolbar">
          <div>
            <h2>Top-10 Feature Form</h2>
            <p>Use sample values for demonstration, then replace them with calculated feature values for a new case.</p>
          </div>
          <div className="feature-form-actions">
            <label>
              Example Row
              <select value={sampleIndex} onChange={(event) => setSampleIndex(Number(event.target.value))}>
                {samples?.rows.map((row, index) => (
                  <option key={row.row_index} value={index}>
                    {row.row_index + 1}
                  </option>
                ))}
              </select>
            </label>
            <button className="secondary-button" onClick={resetToSample}>
              <RotateCcw size={16} />
              Use Sample Values
            </button>
            <button className="secondary-button" onClick={clearInputs}>
              Clear
            </button>
          </div>
        </div>

        <div className="manual-feature-grid">
          {featureNames.map((feature, index) => {
            const info = FEATURE_INFO[feature] ?? {
              label: feature,
              unit: "numeric value",
              meaning: "Speech feature value used by the trained model.",
            };
            const value = feature in edits ? edits[feature] : currentSample?.features[feature];
            return (
              <label className="manual-feature-card" key={feature}>
                <span className="feature-number">{index + 1}</span>
                <span className="feature-copy">
                  <strong>{info.label}</strong>
                  <small>{feature}</small>
                  <em>{info.meaning}</em>
                </span>
                <span className="feature-input-wrap">
                  <span>{info.unit}</span>
                  <input
                    type="number"
                    value={value ?? ""}
                    placeholder="Enter value"
                    onChange={(event) =>
                      setEdits((existing) => ({
                        ...existing,
                        [feature]: event.target.value === "" ? null : Number(event.target.value),
                      }))
                    }
                  />
                </span>
              </label>
            );
          })}
        </div>

        <div className="feature-runbar">
          <div className="model-note">
            <CheckCircle2 size={16} />
            <span>{selectedModel?.model_key ?? "No model selected"}</span>
          </div>
          <span className={missingCount ? "input-warning" : "input-ready"}>
            {missingCount ? `${missingCount} value${missingCount === 1 ? "" : "s"} missing` : "All 10 values ready"}
          </span>
          <button className="primary-button" onClick={onPredict} disabled={isPredicting || !canPredict}>
            {isPredicting ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
            Predict From Values
          </button>
        </div>
      </section>
    </div>
  );
}

function hasNumericFeatureValue(
  feature: string,
  edits: Record<string, number | null>,
  currentSample: SampleRow | null,
) {
  const value = feature in edits ? edits[feature] : currentSample?.features[feature];
  return typeof value === "number" && Number.isFinite(value);
}
