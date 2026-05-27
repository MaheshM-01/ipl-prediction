from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MonitoringConfig:
    logs_dir: Path = Path("logs/monitoring")
    service_name: str = "ipl-prediction-api"
    drift_threshold: float = 0.20
    psi_bins: int = 10


class IPLMonitor:
    """Lightweight monitoring utility for inference logging and drift checks."""

    def __init__(self, cfg: MonitoringConfig | None = None) -> None:
        self.cfg = cfg or MonitoringConfig()
        self.logs_dir = Path(self.cfg.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.prediction_log_path = self.logs_dir / "predictions.jsonl"
        self.error_log_path = self.logs_dir / "errors.jsonl"
        self.health_path = self.logs_dir / "health.json"
        self.drift_log_path = self.logs_dir / "drift_reports.jsonl"

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_json(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [IPLMonitor._safe_json(v) for v in value]
        if isinstance(value, dict):
            return {str(k): IPLMonitor._safe_json(v) for k, v in value.items()}
        return str(value)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")

    def log_prediction(
        self,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any],
        *,
        latency_ms: float | None = None,
        status_code: int = 200,
    ) -> None:
        row = {
            "timestamp_utc": self._utc_now_iso(),
            "service": self.cfg.service_name,
            "type": "prediction",
            "status_code": status_code,
            "latency_ms": float(latency_ms) if latency_ms is not None else None,
            "request": self._safe_json(dict(request_payload)),
            "response": self._safe_json(dict(response_payload)),
        }
        self._append_jsonl(self.prediction_log_path, row)

    def log_error(
        self,
        error_message: str,
        *,
        request_payload: Mapping[str, Any] | None = None,
        stage: str = "inference",
    ) -> None:
        row = {
            "timestamp_utc": self._utc_now_iso(),
            "service": self.cfg.service_name,
            "type": "error",
            "stage": stage,
            "error": str(error_message),
            "request": self._safe_json(dict(request_payload or {})),
        }
        self._append_jsonl(self.error_log_path, row)

    def update_health(
        self,
        *,
        status: str,
        models_loaded: list[str],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "timestamp_utc": self._utc_now_iso(),
            "service": self.cfg.service_name,
            "status": status,
            "models_loaded": models_loaded,
            "extra": self._safe_json(dict(extra or {})),
        }
        with self.health_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    @staticmethod
    def _psi_for_series(
        reference: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        ref = reference.dropna().astype(float)
        cur = current.dropna().astype(float)

        if ref.empty or cur.empty:
            return 0.0

        if np.isclose(ref.nunique(), 1) and np.isclose(cur.nunique(), 1):
            return 0.0

        try:
            quantiles = np.linspace(0, 1, bins + 1)
            edges = np.unique(np.quantile(ref, quantiles))
            if len(edges) < 3:
                return 0.0

            ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
            cur_bins = pd.cut(cur, bins=edges, include_lowest=True)

            ref_pct = ref_bins.value_counts(normalize=True).sort_index()
            cur_pct = cur_bins.value_counts(normalize=True).reindex(ref_pct.index).fillna(0.0)

            eps = 1e-6
            ref_pct = np.clip(ref_pct.values, eps, None)
            cur_pct = np.clip(cur_pct.values, eps, None)

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return float(psi)
        except Exception:
            return 0.0

    def feature_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        *,
        numeric_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        if numeric_cols is None:
            numeric_cols = [
                col
                for col in reference_df.columns
                if col in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[col])
            ]

        per_feature = {}
        drifted = []

        for col in numeric_cols:
            psi_value = self._psi_for_series(
                reference_df[col], current_df[col], bins=self.cfg.psi_bins
            )
            is_drifted = psi_value >= self.cfg.drift_threshold
            per_feature[col] = {"psi": round(psi_value, 6), "drifted": is_drifted}
            if is_drifted:
                drifted.append(col)

        report = {
            "timestamp_utc": self._utc_now_iso(),
            "service": self.cfg.service_name,
            "drift_threshold": self.cfg.drift_threshold,
            "feature_count_checked": len(numeric_cols),
            "drifted_features": drifted,
            "has_drift": len(drifted) > 0,
            "features": per_feature,
        }

        self._append_jsonl(self.drift_log_path, report)
        return report

    def daily_summary(self) -> dict[str, Any]:
        total_predictions = 0
        success_predictions = 0
        total_latency = 0.0
        errors = 0

        if self.prediction_log_path.exists():
            for line in self.prediction_log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                total_predictions += 1
                if int(row.get("status_code", 500)) < 400:
                    success_predictions += 1
                latency = row.get("latency_ms")
                if latency is not None:
                    total_latency += float(latency)

        if self.error_log_path.exists():
            for line in self.error_log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    errors += 1

        avg_latency = (
            round(total_latency / total_predictions, 3) if total_predictions > 0 else None
        )

        return {
            "timestamp_utc": self._utc_now_iso(),
            "service": self.cfg.service_name,
            "total_predictions": total_predictions,
            "successful_predictions": success_predictions,
            "total_errors": errors,
            "avg_latency_ms": avg_latency,
        }
