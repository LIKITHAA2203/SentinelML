/**
 * SentinelML API Configuration
 * Centralized, production-ready configuration for the deployed FastAPI backend.
 */

// Production deployed FastAPI backend on Render
export const DEFAULT_API_BASE_URL = 'https://sentinelml-api-od5n.onrender.com';

// Configurable API base URL: defaults strictly to the deployed Render URL
export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, '');

// Official FastAPI endpoints defined in src/sentinelml/api/main.py
export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/health`,
  MODEL_INFO: `${API_BASE_URL}/model-info`,
  METRICS: `${API_BASE_URL}/metrics`,
  PREDICT: `${API_BASE_URL}/predict`,
  DRIFT: `${API_BASE_URL}/monitoring/drift`,
} as const;

export interface TransactionPayload {
  Time: number;
  Amount: number;
  V1: number;
  V2: number;
  V3: number;
  V4: number;
  V5: number;
  V6: number;
  V7: number;
  V8: number;
  V9: number;
  V10: number;
  V11: number;
  V12: number;
  V13: number;
  V14: number;
  V15: number;
  V16: number;
  V17: number;
  V18: number;
  V19: number;
  V20: number;
  V21: number;
  V22: number;
  V23: number;
  V24: number;
  V25: number;
  V26: number;
  V27: number;
  V28: number;
}

export interface PredictionApiResponse {
  request_id: string;
  model_version: string;
  risk_score: number;
  decision_threshold: number;
  predicted_label: number;
  classification: string;
  latency_ms: number;
  status: string;
  disclaimer?: string;
}

export interface HealthApiResponse {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  database_connected: boolean;
}

export interface MetricsApiResponse {
  total_requests: number;
  predicted_frauds: number;
  predicted_fraud_rate_pct: number;
  avg_latency_ms: number;
  avg_risk_score: number;
}

export interface ModelInfoApiResponse {
  model_name: string;
  model_version: string;
  algorithm: string;
  decision_threshold: number;
  feature_count: number;
  training_timestamp: string;
  offline_metrics?: {
    validation?: Record<string, number>;
    test?: Record<string, number>;
  };
}

export interface DriftApiResponse {
  overall_status?: string;
  status?: string;
  message?: string;
  monitored_features_count?: number;
  drifted_features_count?: number;
  features?: Record<string, any> | Array<any>;
  notice?: string;
}
