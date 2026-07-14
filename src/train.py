"""
train.py
Train baseline (Logistic Regression) + comparator models (Random Forest,
XGBoost) on a TEMPORAL split (no shuffling -> no leakage), evaluate with
log loss / Brier score, pick the best-calibrated model, and persist a single
joblib artifact (model + scaler) plus metadata.json.

Model is trained ONCE on historical data. It is NOT retrained when new
World Cup 2026 results come in -- those only feed live inference features
(see src/features.py:build_live_feature_row).
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_COLUMNS
from src.evaluate import evaluate_probs

SPLIT_DATE = "2018-01-01"
MODEL_VERSION = "wc2026-knockout-v1"


def load_features() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "training_features.csv", parse_dates=["date"])


def temporal_split(df: pd.DataFrame):
    train = df[df["date"] < SPLIT_DATE]
    test = df[df["date"] >= SPLIT_DATE]
    return train, test


def fit_and_score(name, model, X_train, y_train, X_test, y_test, calibrate=False):
    if calibrate:
        model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_probs(y_test, probs)
    print(f"[{name}] {metrics}")
    return model, metrics


def main():
    df = load_features()
    train, test = temporal_split(df)
    X_train, y_train = train[FEATURE_COLUMNS], train["target"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target"]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # 1. Baseline: Logistic Regression (wajib)
    logreg = LogisticRegression(max_iter=1000)
    logreg, m1 = fit_and_score("LogisticRegression (baseline)", logreg, X_train_s, y_train, X_test_s, y_test)
    results["logistic_regression"] = {"model": logreg, "metrics": m1, "uses_scaler": True}

    # 2. Comparator: Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    rf, m2 = fit_and_score("RandomForest (comparator)", rf, X_train.values, y_train, X_test.values, y_test)
    results["random_forest"] = {"model": rf, "metrics": m2, "uses_scaler": False}

    # 3. Optional: XGBoost comparator
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
        )
        xgb, m3 = fit_and_score("XGBoost (comparator)", xgb, X_train.values, y_train, X_test.values, y_test)
        results["xgboost"] = {"model": xgb, "metrics": m3, "uses_scaler": False}
    except ImportError:
        print("xgboost not installed, skipping optional comparator")

    # Pick best model by log loss (probability quality is the primary metric)
    best_name = min(results, key=lambda k: results[k]["metrics"]["log_loss"])
    best = results[best_name]
    print(f"\nSelected model: {best_name} (log_loss={best['metrics']['log_loss']:.4f})")

    # Calibrate the chosen model if it looks overconfident (Brier notably > log_loss/4 heuristic
    # or probabilities are extreme) -- always safe to calibrate on the training fold via CV.
    final_model = best["model"]
    X_train_final = X_train_s if best["uses_scaler"] else X_train.values
    X_test_final = X_test_s if best["uses_scaler"] else X_test.values

    prob_std = np.std(final_model.predict_proba(X_test_final)[:, 1])
    calibrated = False
    if best["metrics"]["brier_score"] > 0.22:
        print("Probabilities look overconfident -> applying CalibratedClassifierCV")
        base_estimator = type(final_model)(**final_model.get_params()) if not hasattr(final_model, "estimator") else final_model
        final_model, cal_metrics = fit_and_score(
            f"{best_name} (calibrated)", base_estimator,
            X_train_final, y_train, X_test_final, y_test, calibrate=True,
        )
        best["metrics"] = cal_metrics
        calibrated = True

    # Persist single artifact: model + scaler + which features need scaling
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    artifact = {
        "model": final_model,
        "scaler": scaler if best["uses_scaler"] else None,
        "uses_scaler": best["uses_scaler"],
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(artifact, models_dir / "model.joblib")

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": best_name,
        "calibrated": calibrated,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "split_date": SPLIT_DATE,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features_used": FEATURE_COLUMNS,
        "backtest_metrics": best["metrics"],
        "all_models_backtest": {k: v["metrics"] for k, v in results.items()},
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model.joblib and metadata.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
