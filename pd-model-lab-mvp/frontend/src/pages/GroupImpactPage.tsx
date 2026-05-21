import type { RefObject } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { GroupImpact } from "../types";

type GroupChartPoint = {
  threshold: number;
  "Healthy Control": number;
  "Parkinson's (PD)": number;
};

type GroupImpactPageProps = {
  refProp?: RefObject<HTMLElement>;
  chartData: GroupChartPoint[];
  threshold: number;
  setThreshold: (threshold: number) => void;
  thresholdRows: GroupImpact["series"];
};

function formatPercent(value?: number | null) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "--";
}

export function GroupImpactPage({
  refProp,
  chartData,
  threshold,
  setThreshold,
  thresholdRows,
}: GroupImpactPageProps) {
  return (
    <section ref={refProp} className="panel chart-panel wide" id="group-impact">
      <div className="section-heading compact">
        <div>
          <h2>Group / Class Impact</h2>
          <p>Positive-rate shifts across decision thresholds.</p>
        </div>
        <label className="threshold-control">
          Threshold {threshold.toFixed(2)}
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>
      </div>
      <div className="impact-layout">
        <ResponsiveContainer width="100%" height={205}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#e8edf1" />
            <XAxis dataKey="threshold" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
            <Tooltip formatter={(value: number) => value.toFixed(3)} />
            <Legend />
            <Line type="monotone" dataKey="Healthy Control" stroke="#0f8b8d" strokeWidth={2.4} dot={false} />
            <Line type="monotone" dataKey="Parkinson's (PD)" stroke="#d98a21" strokeWidth={2.4} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="impact-summary">
          <h3>Threshold Summary</h3>
          {thresholdRows.map((row) => (
            <div className="summary-row" key={row.group}>
              <span>{row.group}</span>
              <strong>{formatPercent(row.positive_rate)}</strong>
              <small>N={row.n}</small>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
