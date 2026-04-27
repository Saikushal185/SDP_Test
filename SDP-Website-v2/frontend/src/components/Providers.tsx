"use client";
import { ReactNode } from "react";
import { PredictionProvider } from "@/context/PredictionContext";

export default function Providers({ children }: { children: ReactNode }) {
  return <PredictionProvider>{children}</PredictionProvider>;
}
