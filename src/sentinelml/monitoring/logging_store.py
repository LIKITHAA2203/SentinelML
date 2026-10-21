"""
SentinelML SQLite Logging Store
Safely persists incoming inference requests, risk scores, decisions, and feature
snapshots for statistical drift monitoring and operational audits.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from sentinelml.config import MONITORING_DB_PATH

logger = logging.getLogger(__name__)


class PredictionStore:
    def __init__(self, db_path: Path = MONITORING_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema with parameterized, safe DDL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Predictions log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    decision_threshold REAL NOT NULL,
                    predicted_label INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    feature_payload_json TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)

            # Explicitly separate table for verified ground-truth labels
            # (Predictions are never treated as ground truth)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verified_outcomes (
                    request_id TEXT PRIMARY KEY,
                    verified_label INTEGER NOT NULL,
                    verified_at TEXT NOT NULL,
                    verified_by TEXT,
                    is_simulated INTEGER DEFAULT 0,
                    FOREIGN KEY(request_id) REFERENCES prediction_logs(request_id)
                )
            """)

            # Drift audit history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_reports (
                    report_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    reference_window_size INTEGER NOT NULL,
                    current_window_size INTEGER NOT NULL,
                    drifted_features_count INTEGER NOT NULL,
                    overall_status TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_prediction(
        self,
        request_id: str,
        model_version: str,
        risk_score: float,
        decision_threshold: float,
        predicted_label: int,
        latency_ms: float,
        feature_payload: Dict[str, float],
        status: str = "SUCCESS"
    ) -> None:
        """Records an inference event with parameterized query."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO prediction_logs 
                (request_id, timestamp, model_version, risk_score, decision_threshold, 
                 predicted_label, latency_ms, feature_payload_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    now_utc,
                    model_version,
                    round(risk_score, 4),
                    round(decision_threshold, 4),
                    predicted_label,
                    round(latency_ms, 2),
                    json.dumps(feature_payload),
                    status,
                )
            )
            conn.commit()

    def get_recent_predictions(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetch latest prediction rows for drift analysis and dashboard display."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT request_id, timestamp, model_version, risk_score, 
                       decision_threshold, predicted_label, latency_ms, 
                       feature_payload_json, status
                FROM prediction_logs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_operational_metrics(self) -> Dict[str, Any]:
        """Aggregate operational stats: count, fraud count, average latency."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as predicted_frauds,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(risk_score) as avg_risk_score
                FROM prediction_logs
            """)
            row = cursor.fetchone()
            total = row["total_requests"] if row and row["total_requests"] else 0
            frauds = row["predicted_frauds"] if row and row["predicted_frauds"] else 0
            fraud_rate = (frauds / total * 100.0) if total > 0 else 0.0
            avg_lat = row["avg_latency_ms"] if row and row["avg_latency_ms"] else 0.0

            return {
                "total_requests": total,
                "predicted_frauds": frauds,
                "predicted_fraud_rate_pct": round(fraud_rate, 3),
                "avg_latency_ms": round(avg_lat, 2),
                "avg_risk_score": round(row["avg_risk_score"] or 0.0, 4)
            }


if __name__ == "__main__":
    store = PredictionStore()
    print("Database initialized successfully at:", store.db_path)
    print("Metrics:", store.get_operational_metrics())
