import type { RefObject } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureImportance } from "../types";

type FeatureImportancePageProps = {
  refProp?: RefObject<HTMLElement>;
  features: FeatureImportance[];
};

function shortFeatureName(value: string) {
  return value.length > 24 ? `${value.slice(0, 21)}...` : value;
}

export function FeatureImportancePage({ refProp, features }: FeatureImportancePageProps) {
  return (
    <section ref={refProp} className="panel chart-panel" id="feature-importance">
      <div className="section-heading compact">
        <div>
          <h2>Feature Importance</h2>
          <p>Top drivers for the selected model.</p>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={205}>
        <BarChart data={features} layout="vertical" margin={{ left: 18, right: 24 }}>
          <CartesianGrid stroke="#e8edf1" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis
            dataKey="feature"
            type="category"
            width={155}
            tick={{ fontSize: 11 }}
            tickFormatter={shortFeatureName}
          />
          <Tooltip formatter={(value: number) => value.toFixed(4)} />
          <Bar dataKey="importance" fill="#139a9a" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}
