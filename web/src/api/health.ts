import type { InferenceHealth, StatsResponse } from "../types";
import { inference } from "./client";

export function getHealth() {
  return inference<InferenceHealth>("/health");
}

export function getStats() {
  return inference<StatsResponse>("/stats");
}
