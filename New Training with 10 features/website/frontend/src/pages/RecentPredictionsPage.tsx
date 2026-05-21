import { Download } from "lucide-react";
import type { PredictionRow } from "../types";

type RecentPredictionsPageProps = {
  predictions: PredictionRow[];
};

export function RecentPredictionsPage({ predictions }: RecentPredictionsPageProps) {
  return (
    <section className="panel recent-panel">
      <div className="section-heading compact">
        <div>
          <h2>Recent Predictions</h2>
          <p>Latest local results from CSV or edited sample rows.</p>
        </div>
        <button className="secondary-button">
          <Download size={16} />
          Download CSV
        </button>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Row</th>
              <th>PD Probability</th>
              <th>Predicted Label</th>
              <th>Model</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((row, index) => (
              <tr key={`${row.model_key}-${index}`}>
                <td>{row.source}</td>
                <td>{row.row_index + 1}</td>
                <td>{row.probability.toFixed(3)}</td>
                <td>{row.predicted_label}</td>
                <td>{row.model_key}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
