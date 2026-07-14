"""
predict.py
Loads the trained model artifact ONCE and serves live predictions by
building a fresh feature row (see features.build_live_feature_row) from
historical + current World Cup 2026 data. Never retrains here.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_live_feature_row, FEATURE_COLUMNS
from src.clean import canonicalize


class KnockoutPredictor:
    def __init__(self):
        artifact = joblib.load(ROOT / "models" / "model.joblib")
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.uses_scaler = artifact["uses_scaler"]
        self.feature_columns = artifact["feature_columns"]

        with open(ROOT / "models" / "metadata.json") as f:
            self.metadata = json.load(f)

        processed = ROOT / "data" / "processed"
        self.matches = pd.read_csv(processed / "processed_matches.csv", parse_dates=["date"])
        self.rankings = pd.read_csv(processed / "processed_rankings.csv", parse_dates=["date"])
        self.wc2026 = pd.read_csv(processed / "processed_wc2026.csv", parse_dates=["date"])

        self.known_teams = sorted(set(self.matches["home_team"]) | set(self.matches["away_team"]))

    def known_team(self, name: str) -> bool:
        return canonicalize(name) in self.known_teams

    def predict(self, team_a: str, team_b: str) -> dict:
        team_a_c = canonicalize(team_a)
        team_b_c = canonicalize(team_b)

        feat = build_live_feature_row(team_a_c, team_b_c, self.matches, self.rankings, self.wc2026)
        X = pd.DataFrame([[feat[c] for c in self.feature_columns]], columns=self.feature_columns)
        if self.uses_scaler and self.scaler is not None:
            X = self.scaler.transform(X)
        else:
            X = X.values

        win_a = float(self.model.predict_proba(X)[0, 1])
        win_b = 1.0 - win_a

        margin = abs(win_a - 0.5)
        if margin < 0.05:
            confidence = "low"
        elif margin < 0.15:
            confidence = "medium"
        else:
            confidence = "high"

        return {
            "team_a": team_a,
            "team_b": team_b,
            "win_a": round(win_a, 4),
            "win_b": round(win_b, 4),
            "model_version": self.metadata["model_version"],
            "confidence": confidence,
            "features_used": self.feature_columns,
        }
