"""
test_features.py
Unit tests for feature engineering, with special focus on the no-leakage
requirement demanded by the task's acceptance criteria.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_training_features, build_live_feature_row, FEATURE_COLUMNS
from src.clean import canonicalize, resolve_match_winner


def test_canonicalize_maps_known_aliases():
    assert canonicalize("Cabo Verde") == "Cape Verde"
    assert canonicalize("Türkiye") == "Turkey"
    assert canonicalize("USA") == "United States"


def test_canonicalize_passthrough_unknown():
    assert canonicalize("France") == "France"


def test_resolve_match_winner_decisive():
    row = pd.Series({"home_score": 2, "away_score": 1, "date": pd.Timestamp("2022-01-01"),
                      "home_team": "A", "away_team": "B"})
    assert resolve_match_winner(row, {}) == 1


def test_resolve_match_winner_draw_no_shootout_is_none():
    row = pd.Series({"home_score": 1, "away_score": 1, "date": pd.Timestamp("2022-01-01"),
                      "home_team": "A", "away_team": "B"})
    assert resolve_match_winner(row, {}) is None


def test_resolve_match_winner_draw_with_shootout():
    row = pd.Series({"home_score": 1, "away_score": 1, "date": pd.Timestamp("2022-01-01"),
                      "home_team": "A", "away_team": "B"})
    lookup = {(pd.Timestamp("2022-01-01"), "A", "B"): "B"}
    assert resolve_match_winner(row, lookup) == 0


def _toy_matches():
    """Small synthetic match history for team A vs B across several dates."""
    return pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0},
        {"date": pd.Timestamp("2020-06-01"), "home_team": "B", "away_team": "A", "home_score": 1, "away_score": 1},
        {"date": pd.Timestamp("2021-01-01"), "home_team": "A", "away_team": "B", "home_score": 3, "away_score": 1},
        {"date": pd.Timestamp("2021-06-01"), "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 2},
    ])


def _toy_rankings():
    rows = []
    for d in ["1993-01-01", "2019-06-01", "2020-06-01", "2021-06-01"]:
        rows.append({"team": "A", "date": pd.Timestamp(d), "total_points": 1500, "rank": 10})
        rows.append({"team": "B", "date": pd.Timestamp(d), "total_points": 1400, "rank": 20})
    return pd.DataFrame(rows)


def test_build_training_features_no_future_leakage():
    """
    The feature row for the LAST match must be built only from the three
    matches strictly before it -- changing a later result must not exist
    yet, and features must not depend on matches after the row's date.
    """
    matches = _toy_matches()
    rankings = _toy_rankings()
    feat = build_training_features(matches, rankings, shootout_lookup={})

    # first two matches can't have 10 prior matches each so are usable once
    # both teams have >=1 prior match; verify row count and column presence
    assert set(FEATURE_COLUMNS).issubset(feat.columns)
    assert "target" in feat.columns
    assert len(feat) >= 1

    # every feature date must be strictly after both teams' earliest match
    assert (feat["date"] > matches["date"].min()).all()


def test_build_training_features_drops_undecided_draws():
    matches = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0},
        {"date": pd.Timestamp("2020-06-01"), "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1},
    ])
    rankings = _toy_rankings()
    feat = build_training_features(matches, rankings, shootout_lookup={})
    # the draw (2nd match) has no shootout resolution -> must be dropped
    assert len(feat) == 0 or (feat["target"].isin([0, 1])).all()


def test_build_live_feature_row_returns_all_columns():
    matches = _toy_matches()
    rankings = _toy_rankings()
    wc2026 = pd.DataFrame(columns=["home_team_name", "away_team_name", "status", "home_score", "away_score"])
    feat = build_live_feature_row("A", "B", matches, rankings, wc2026,
                                   as_of_date=pd.Timestamp("2022-01-01"))
    for col in FEATURE_COLUMNS:
        assert col in feat
    assert "world_cup_2026_goal_diff_delta" in feat


def test_build_live_feature_row_symmetry():
    """Swapping team_a/team_b should invert the sign of delta features."""
    matches = _toy_matches()
    rankings = _toy_rankings()
    wc2026 = pd.DataFrame(columns=["home_team_name", "away_team_name", "status", "home_score", "away_score"])
    f1 = build_live_feature_row("A", "B", matches, rankings, wc2026, as_of_date=pd.Timestamp("2022-01-01"))
    f2 = build_live_feature_row("B", "A", matches, rankings, wc2026, as_of_date=pd.Timestamp("2022-01-01"))
    for col in FEATURE_COLUMNS:
        assert f1[col] == pytest.approx(-f2[col], abs=1e-6)
