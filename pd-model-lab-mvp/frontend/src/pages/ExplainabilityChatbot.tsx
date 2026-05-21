import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Bot, Send, UserRound } from "lucide-react";
import type { GroupedExplanation, ModelInfo, PredictionRow } from "../types";

type ChatMessage = {
  role: "assistant" | "user";
  text: string;
};

const PROMPTS = [
  "Why this prediction?",
  "What increased the score?",
  "What lowered the score?",
  "What does baseline mean?",
  "Can this diagnose PD?",
];

function formatPrecisePercent(value?: number | null) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "--";
}

function formatContribution(value?: number | null) {
  if (typeof value !== "number") return "--";
  const points = Math.abs(value) * 100;
  const digits = points >= 10 ? 1 : 2;
  return `${value >= 0 ? "+" : "-"}${points.toFixed(digits)} percentage points`;
}

function methodLabel(method?: string) {
  if (method === "native") return "Native SHAP";
  if (method === "kernel-grouped") return "Grouped Kernel SHAP";
  return "SHAP";
}

function groupList(groups: GroupedExplanation[], emptyText: string) {
  if (!groups.length) return emptyText;
  return groups
    .slice(0, 3)
    .map((group) => `${group.name} (${formatContribution(group.value)}, ${group.featureCount} features)`)
    .join("; ");
}

function predictionSignature(prediction: PredictionRow | null) {
  if (!prediction) return "empty";
  return [
    prediction.model_key,
    prediction.row_index,
    prediction.source,
    prediction.probability.toFixed(6),
  ].join("|");
}

function buildStandardAnswer(question: string) {
  const normalized = question.toLowerCase();

  if (/(diagnos|doctor|clinical|medical|disease|safe|trust|can this)/i.test(normalized)) {
    return "No. This tool cannot diagnose Parkinson's disease. It only explains a machine-learning prediction and should be used as research-support information, not medical advice.";
  }

  if (/(baseline|base value|base_value|grouped shift|shift)/i.test(normalized)) {
    return "The baseline is the model's starting PD probability before the current sample's voice features move the score up or down. Grouped shift is the total movement from that baseline to the final displayed probability.";
  }

  if (/(increase|raised|higher|positive|red|pushed up|score up)/i.test(normalized)) {
    return "The score increases when grouped voice features push the model toward a Parkinson's-positive prediction. In this page, those upward effects are shown with red bars after a prediction is run.";
  }

  if (/(decrease|lower|lowered|green|reduced|softened|score down)/i.test(normalized)) {
    return "The score decreases when grouped voice features push the model away from a Parkinson's-positive prediction. In this page, those downward effects are shown with green bars after a prediction is run.";
  }

  if (/(probability|confidence|risk|prediction|why|result|label)/i.test(normalized)) {
    return "A prediction combines the model's baseline probability with the sample's grouped SHAP movements. After you run a sample, this chat will state the exact predicted label, probability, confidence, baseline, and strongest drivers.";
  }

  if (/(shap|method|kernel|native|explain|xai)/i.test(normalized)) {
    return "SHAP is an explainability method that estimates how each feature or feature group changes the model's output. This page groups related voice measurements so the explanation is easier to read.";
  }

  return "This chat can explain the prediction, probability, confidence, baseline, SHAP method, and which voice-signal groups increase or decrease the PD score. Run a prediction for exact sample-specific values.";
}

function buildAnswer(question: string, prediction: PredictionRow | null, selectedModel: ModelInfo | null) {
  if (!prediction?.explanation) return buildStandardAnswer(question);

  const normalized = question.toLowerCase();
  const explanation = prediction.explanation;
  const groups = explanation?.groups ?? [];
  const increased = groups.filter((group) => group.value > 0).sort((a, b) => b.absValue - a.absValue);
  const decreased = groups.filter((group) => group.value < 0).sort((a, b) => b.absValue - a.absValue);
  const baseValue = typeof explanation?.base_value === "number" ? explanation.base_value : null;
  const groupedShift = baseValue !== null ? prediction.probability - baseValue : null;
  const modelName =
    selectedModel?.model_key === prediction.model_key
      ? selectedModel.model_name
      : prediction.model_key.split("_").pop() ?? prediction.model_key;
  const method = methodLabel(explanation?.method);
  const probability = formatPrecisePercent(prediction.probability);
  const confidence = formatPrecisePercent(prediction.confidence);
  const baseline = formatPrecisePercent(baseValue);
  const shift = formatContribution(groupedShift);

  if (/(diagnos|doctor|clinical|medical|disease|safe|trust|can this)/i.test(normalized)) {
    return "No. This is research-support output only. It can explain how the model reached this score for the uploaded sample, but it cannot diagnose Parkinson's disease or replace clinical review.";
  }

  if (/(baseline|base value|base_value|grouped shift|shift)/i.test(normalized)) {
    if (baseValue === null) {
      return `The model did not return a baseline value for this explanation. The displayed probability is ${probability}, and the available grouped SHAP drivers show the direction of influence.`;
    }
    return `Baseline is the model's starting Parkinson's-positive probability before this sample's grouped voice-signal changes are added. Here the baseline is ${baseline}, the grouped shift is ${shift}, and the final displayed probability is ${probability}.`;
  }

  if (/(probability|confidence|risk|prediction|why|result|label)/i.test(normalized)) {
    const topUp = groupList(increased, "no upward grouped driver");
    const topDown = groupList(decreased, "no downward grouped driver");
    return `The ${modelName} model predicted ${prediction.predicted_label} with ${probability} probability and ${confidence} confidence. The final probability comes from the baseline ${baseline}, plus a grouped shift of ${shift}. Main upward drivers: ${topUp}. Main downward drivers: ${topDown}.`;
  }

  if (/(increase|raised|higher|positive|red|pushed up|score up)/i.test(normalized)) {
    return `The strongest groups that increased the Parkinson's-positive probability were: ${groupList(increased, "none of the grouped drivers increased the score")}. Red bars in the chart show these upward contributions.`;
  }

  if (/(decrease|lower|lowered|green|reduced|softened|score down)/i.test(normalized)) {
    return `The strongest groups that lowered the Parkinson's-positive probability were: ${groupList(decreased, "none of the grouped drivers lowered the score")}. Green bars in the chart show these downward contributions.`;
  }

  if (/(shap|method|kernel|native|explain|xai)/i.test(normalized)) {
    return `This explanation uses ${method} on the ${explanation?.output_scale ?? "probability"} scale. The groups summarize many raw voice features into readable signal families, so the chart shows grouped percentage-point movements rather than a raw 753-feature table.`;
  }

  return "I can answer questions about this prediction's probability, confidence, baseline, grouped shift, SHAP method, and which grouped voice-signal families increased or lowered the score. Try one of the prompt chips above.";
}

export function ExplainabilityChatbot({
  prediction,
  selectedModel,
}: {
  prediction: PredictionRow | null;
  selectedModel: ModelInfo | null;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const signature = useMemo(() => predictionSignature(prediction), [prediction]);
  const disabled = !prediction?.explanation;

  useEffect(() => {
    if (!prediction?.explanation) {
      setMessages([
        {
          role: "assistant",
          text: "Ask me about the explanation terms. After you run a prediction, I will answer with exact values for that sample.",
        },
      ]);
      setDraft("");
      return;
    }

    setMessages([
      {
        role: "assistant",
        text: "Ask me about this prediction's probability, baseline, grouped SHAP drivers, or what the red and green bars mean.",
      },
    ]);
    setDraft("");
  }, [signature, prediction]);

  function ask(question: string) {
    const cleanQuestion = question.trim();
    if (!cleanQuestion) return;

    const answer = buildAnswer(cleanQuestion, prediction, selectedModel);
    setMessages((existing) => [
      ...existing,
      { role: "user", text: cleanQuestion },
      { role: "assistant", text: answer },
    ]);
    setDraft("");
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    ask(draft);
  }

  return (
    <div className={`xai-chatbot ${disabled ? "disabled" : ""}`}>
      <div className="xai-chatbot-header">
        <div>
          <span>XAI Chat</span>
          <h3>Ask about this explanation</h3>
        </div>
        <Bot size={22} />
      </div>

      <div className="chat-prompts" aria-label="Suggested XAI questions">
        {PROMPTS.map((prompt) => (
          <button key={prompt} type="button" onClick={() => ask(prompt)}>
            {prompt}
          </button>
        ))}
      </div>

      <div className="chat-thread" aria-live="polite">
        {messages.map((message, index) => (
          <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
            {message.role === "assistant" ? <Bot size={16} /> : <UserRound size={16} />}
            <p>{message.text}</p>
          </div>
        ))}
      </div>

      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          value={draft}
          placeholder={disabled ? "Ask a general XAI question" : "Ask about this prediction"}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button type="submit" disabled={!draft.trim()} aria-label="Send chat message">
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}
