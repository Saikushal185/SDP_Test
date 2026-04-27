"use client";

interface ProbabilityGaugeProps {
  probability: number;
}

export default function ProbabilityGauge({ probability }: ProbabilityGaugeProps) {
  const percentage = Math.round(probability * 100);
  const angle = probability * 180;
  const riskLevel =
    probability >= 0.7 ? "High" : probability >= 0.4 ? "Medium" : "Low";
  const riskColor =
    probability >= 0.7
      ? "text-[var(--danger)]"
      : probability >= 0.4
      ? "text-[var(--warning)]"
      : "text-[var(--success)]";

  const strokeColor =
    probability >= 0.7
      ? "#8e4b43"
      : probability >= 0.4
      ? "#8d6234"
      : "#2f6a55";

  const radius = 80;
  const circumference = Math.PI * radius;
  const dashOffset = circumference - probability * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-52 h-28 overflow-hidden">
        <svg viewBox="0 0 200 110" className="w-full h-full">
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="rgba(17, 33, 38, 0.12)"
            strokeWidth="16"
            strokeLinecap="round"
          />
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={strokeColor}
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={`${circumference}`}
            strokeDashoffset={dashOffset}
            style={{
              transition: "stroke-dashoffset 1s ease-out",
            }}
          />
          <line
            x1="100"
            y1="100"
            x2={100 + 60 * Math.cos(Math.PI - (angle * Math.PI) / 180)}
            y2={100 - 60 * Math.sin((angle * Math.PI) / 180)}
            stroke="#173a41"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <circle cx="100" cy="100" r="5" fill="#173a41" />
        </svg>
      </div>
      <div className="text-center mt-2">
        <p className="font-display text-3xl text-[var(--text-strong)]">
          {percentage}%
        </p>
        <p className={`text-sm font-semibold ${riskColor}`}>
          Risk level: {riskLevel}
        </p>
      </div>
    </div>
  );
}
