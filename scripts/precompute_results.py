from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import (  # noqa: E402
    COIN_OPTIONS,
    DEFAULT_DATA_PATH,
    MODEL_OPTIONS,
    load_dataset,
    run_sarimax_prediction,
    run_xgboost_prediction,
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "precomputed_results.pkl"
HORIZONS = [1, 3, 7]
TIMEZONE = ZoneInfo("Asia/Jakarta")


def calculate_confidence(metrics: dict[str, float]) -> tuple[float, str]:
    directional = float(metrics.get("Directional Accuracy", 0) or 0)
    normalized_r2 = max(0.0, min(float(metrics.get("R2", 0) or 0), 1.0)) * 100
    inverse_mape = max(0.0, min(100 - float(metrics.get("MAPE", 100) or 100), 100.0))
    confidence = (directional * 0.4) + (normalized_r2 * 0.3) + (inverse_mape * 0.3)
    if confidence >= 80:
        level = "High"
    elif confidence >= 60:
        level = "Medium"
    else:
        level = "Low"
    return confidence, level


def build_comparison_result(coin_name: str, predictions: dict[tuple[str, str, int], dict[str, object]]) -> dict[str, object]:
    results = {
        model_name: predictions[(coin_name, model_name, 7)]
        for model_name in MODEL_OPTIONS
    }
    rows = []
    for model_name, result in results.items():
        metrics = result["metrics"]
        confidence_score, confidence_level = calculate_confidence(metrics)
        rows.append(
            {
                "Koin": COIN_OPTIONS[coin_name]["label"],
                "Model": model_name,
                "RMSE": metrics.get("RMSE"),
                "MAE": metrics.get("MAE"),
                "MAPE": metrics.get("MAPE"),
                "R2": metrics.get("R2"),
                "Directional Accuracy": metrics.get("Directional Accuracy"),
                "Cumulative Return": metrics.get("Cumulative Return"),
                "Confidence Score": confidence_score,
                "Reliability": confidence_level,
            }
        )

    comparison_table = pd.DataFrame(rows)
    best_row = (
        comparison_table.sort_values(
            by=["Confidence Score", "MAPE", "RMSE"],
            ascending=[False, True, True],
        )
        .iloc[0]
        .to_dict()
    )
    return {
        "coin_name": coin_name,
        "results": results,
        "comparison_table": comparison_table,
        "best_row": best_row,
        "test_period": results["SARIMAX"].get("test_period"),
    }


def main() -> None:
    os.environ.setdefault("FAST_SARIMAX", "1")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DEFAULT_DATA_PATH)
    predictions: dict[tuple[str, str, int], dict[str, object]] = {}

    for coin_name in COIN_OPTIONS:
        for days in HORIZONS:
            print(f"Precomputing SARIMAX | {coin_name} | {days}D", flush=True)
            predictions[(coin_name, "SARIMAX", days)] = run_sarimax_prediction(coin_name, dataset, days=days)

            print(f"Precomputing XGBoost | {coin_name} | {days}D", flush=True)
            predictions[(coin_name, "XGBoost", days)] = run_xgboost_prediction(coin_name, dataset, days=days)

    comparisons = {
        coin_name: build_comparison_result(coin_name, predictions)
        for coin_name in COIN_OPTIONS
    }
    bundle = {
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "data_source": dataset.attrs.get("source", "unknown"),
        "data_range": (
            pd.to_datetime(dataset["Date"]).min(),
            pd.to_datetime(dataset["Date"]).max(),
        ),
        "predictions": predictions,
        "comparisons": comparisons,
    }

    with OUTPUT_PATH.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved precomputed results to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
