import { Fragment } from "react";
import type { CSSProperties } from "react";
import type { ConfusionMatrix } from "../types";

type ConfusionMatrixPageProps = {
  matrix: ConfusionMatrix | null;
};

const LABELS = ["Healthy Control", "Parkinson's (PD)"] as const;

export function ConfusionMatrixPage({ matrix }: ConfusionMatrixPageProps) {
  const maxCount = Math.max(1, ...(matrix?.cells.map((cell) => cell.count) ?? [0]));

  return (
    <section className="panel chart-panel confusion-panel" id="confusion-matrix">
      <div className="section-heading compact">
        <div>
          <h2>Confusion Matrix</h2>
          <p>10-fold validation counts for the selected model.</p>
        </div>
        <div className="dataset-chip">{matrix?.model_name ?? "Loading"}</div>
      </div>

      <div className="confusion-grid" role="table" aria-label="Confusion matrix">
        <div className="confusion-corner" />
        {LABELS.map((label) => (
          <div className="confusion-axis predicted" role="columnheader" key={`predicted-${label}`}>
            Predicted {label}
          </div>
        ))}
        {LABELS.map((actualLabel) => (
          <Fragment key={`row-${actualLabel}`}>
            <div className="confusion-axis actual" role="rowheader" key={`actual-${actualLabel}`}>
              Actual {actualLabel}
            </div>
            {LABELS.map((predictedLabel) => {
              const cell = matrix?.cells.find(
                (item) => item.actual_label === actualLabel && item.predicted_label === predictedLabel,
              );
              const count = cell?.count ?? 0;
              const intensity = count / maxCount;
              return (
                <div
                  className={`confusion-cell ${actualLabel === predictedLabel ? "correct" : "incorrect"}`}
                  role="cell"
                  key={`${actualLabel}-${predictedLabel}`}
                  style={{ "--intensity": intensity } as CSSProperties}
                >
                  <strong>{count}</strong>
                  <span>{actualLabel === predictedLabel ? "Correct" : "Incorrect"}</span>
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </section>
  );
}
