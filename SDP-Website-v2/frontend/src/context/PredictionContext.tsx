"use client";

import { ReactNode, createContext, startTransition, useContext, useState } from "react";
import type { ExplanationResult, PredictionResult } from "@/lib/api";

export interface PredictionBundle {
  prediction: PredictionResult | null;
  explanation: ExplanationResult | null;
  features: Record<string, number> | null;
  model: string | null;
  modelDisplayName: string | null;
  trainingDataset: string | null;
  uploadedFileName: string | null;
}

interface PredictionContextType extends PredictionBundle {
  setPredictionBundle: (bundle: PredictionBundle | null) => void;
  clearPrediction: () => void;
}

const emptyPrediction: PredictionBundle = {
  prediction: null,
  explanation: null,
  features: null,
  model: null,
  modelDisplayName: null,
  trainingDataset: null,
  uploadedFileName: null,
};

const PredictionContext = createContext<PredictionContextType>({
  ...emptyPrediction,
  setPredictionBundle: () => {},
  clearPrediction: () => {},
});

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [bundle, setBundle] = useState<PredictionBundle>(emptyPrediction);

  const setPredictionBundle = (nextBundle: PredictionBundle | null) => {
    startTransition(() => {
      setBundle(nextBundle ?? emptyPrediction);
    });
  };

  const clearPrediction = () => {
    startTransition(() => {
      setBundle(emptyPrediction);
    });
  };

  return (
    <PredictionContext.Provider
      value={{
        ...bundle,
        setPredictionBundle,
        clearPrediction,
      }}
    >
      {children}
    </PredictionContext.Provider>
  );
}

export function usePrediction() {
  return useContext(PredictionContext);
}
