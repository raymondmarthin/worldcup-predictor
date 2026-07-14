"""
features.py
Feature engineering for the knockout win-probability model.

Hard rule (data leakage): every feature for a match must only use information
strictly BEFORE that match's date. Team history is walked chronologically with
a rolling window so no future result ever leaks into a past prediction.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

FEATURE_COLUMNS = [
    "elo_delta",
    "rank_delta",
    "recent_form_delta",
    "goal_diff_recent_delta",
    "rest_days_delta",
]
LIVE_ONLY_FEATURE = "world_cup_2026_goal_diff_delta"
FORM_WINDOW = 10  # last N matches used for recent form / goal diff


def _ranking_lookup(rankings: pd.DataFrame):
    """team -> sorted (date, total_points, rank) tuples, for as-of lookups."""
    lut = defaultdict(list)
    for row in rankings.itertuples(index=False):
        lut[row.team].append((row.date, row.total_points, row.rank))
    for team in lut:
        lut[team].sort(key=lambda t: t[0])
    return lut


def _as_of(lut_list, as_of_date):
    """Most recent (date, points, rank) strictly before as_of_date, else None."""
    result = None
    for d, pts, rk in lut_list:
        if d < as_of_date:
            result = (d, pts, rk)
        else:
            break
    return result


def build_training_features(matches: pd.DataFrame, rankings: pd.DataFrame,
                             shootout_lookup: dict) -> pd.DataFrame:
    """
    Build one training row per match with a decisive winner.
    Only matches from 1993-01-01 onward are used, since FIFA ranking
    coverage (our elo/rank proxy) starts in Dec 1992 -> documented limitation.
    """
    from src.clean import resolve_match_winner

    matches = matches[matches["date"] >= "1993-01-01"].copy()
    rank_lut = _ranking_lookup(rankings)

    # rolling per-team history: deque of (date, goals_for, goals_against, points, match_date)
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    last_played: dict[str, pd.Timestamp] = {}

    rows = []
    matches = matches.sort_values("date").reset_index(drop=True)
    for row in matches.itertuples(index=False):
        home, away, date = row.home_team, row.away_team, row.date

        winner = resolve_match_winner(
            pd.Series({"home_score": row.home_score, "away_score": row.away_score,
                       "date": date, "home_team": home, "away_team": away}),
            shootout_lookup,
        )

        home_hist = history[home]
        away_hist = history[away]

        if winner is not None and len(home_hist) > 0 and len(away_hist) > 0:
            home_rank = _as_of(rank_lut.get(home, []), date)
            away_rank = _as_of(rank_lut.get(away, []), date)
            if home_rank is not None and away_rank is not None:
                elo_delta = home_rank[1] - away_rank[1]
                rank_delta = home_rank[2] - away_rank[2]

                home_form = np.mean([h[2] for h in home_hist])
                away_form = np.mean([h[2] for h in away_hist])
                home_gd = np.mean([h[0] - h[1] for h in home_hist])
                away_gd = np.mean([h[0] - h[1] for h in away_hist])

                home_rest = (date - last_played[home]).days if home in last_played else np.nan
                away_rest = (date - last_played[away]).days if away in last_played else np.nan

                rows.append({
                    "date": date, "home_team": home, "away_team": away,
                    "elo_delta": elo_delta,
                    "rank_delta": rank_delta,
                    "recent_form_delta": home_form - away_form,
                    "goal_diff_recent_delta": home_gd - away_gd,
                    "rest_days_delta": (home_rest - away_rest)
                        if not (np.isnan(home_rest) or np.isnan(away_rest)) else 0.0,
                    "target": winner,
                })

        # update rolling history AFTER using it (no leakage)
        home_pts = 3 if row.home_score > row.away_score else (1 if row.home_score == row.away_score else 0)
        away_pts = 3 if row.away_score > row.home_score else (1 if row.away_score == row.home_score else 0)
        home_hist.append((row.home_score, row.away_score, home_pts))
        away_hist.append((row.away_score, row.home_score, away_pts))
        last_played[home] = date
        last_played[away] = date

    columns = ["date", "home_team", "away_team", *FEATURE_COLUMNS, "target"]
    if not rows:
        return pd.DataFrame(columns=columns)
    feat_df = pd.DataFrame(rows)
    feat_df["rest_days_delta"] = feat_df["rest_days_delta"].fillna(0.0)
    return feat_df


def build_live_feature_row(team_a: str, team_b: str, matches: pd.DataFrame,
                            rankings: pd.DataFrame, wc2026: pd.DataFrame,
                            as_of_date: pd.Timestamp | None = None) -> dict:
    """
    Build a single feature row for a live /predict request, using:
    - historical `matches` (pre-2026) for elo/rank/form/rest as-of features
    - completed `wc2026` matches as extra live context (goal diff so far in
      the 2026 tournament)
    No retraining involved -- this only assembles an inference feature row.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.now("UTC").tz_localize(None)

    rank_lut = _ranking_lookup(rankings)
    a_rank = _as_of(rank_lut.get(team_a, []), as_of_date)
    b_rank = _as_of(rank_lut.get(team_b, []), as_of_date)
    elo_delta = (a_rank[1] - b_rank[1]) if (a_rank and b_rank) else 0.0
    rank_delta = (a_rank[2] - b_rank[2]) if (a_rank and b_rank) else 0.0

    def recent_team_matches(team):
        m = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
        m = m[m["date"] < as_of_date].sort_values("date").tail(FORM_WINDOW)
        return m

    def form_and_gd(team):
        m = recent_team_matches(team)
        if len(m) == 0:
            return 0.0, 0.0
        pts, gds = [], []
        for r in m.itertuples(index=False):
            if r.home_team == team:
                gf, ga = r.home_score, r.away_score
            else:
                gf, ga = r.away_score, r.home_score
            pts.append(3 if gf > ga else (1 if gf == ga else 0))
            gds.append(gf - ga)
        return float(np.mean(pts)), float(np.mean(gds))

    def rest_days(team):
        m = recent_team_matches(team)
        if len(m) == 0:
            return 0.0
        return float((as_of_date - m["date"].max()).days)

    a_form, a_gd = form_and_gd(team_a)
    b_form, b_gd = form_and_gd(team_b)
    rest_delta = rest_days(team_a) - rest_days(team_b)

    def wc2026_gd(team):
        m = wc2026[((wc2026["home_team_name"] == team) | (wc2026["away_team_name"] == team))
                    & (wc2026["status"] == "Completed")]
        if len(m) == 0:
            return 0.0
        gds = []
        for r in m.itertuples(index=False):
            if r.home_team_name == team:
                gds.append(r.home_score - r.away_score)
            else:
                gds.append(r.away_score - r.home_score)
        return float(np.mean(gds))

    wc_gd_delta = wc2026_gd(team_a) - wc2026_gd(team_b)

    return {
        "elo_delta": elo_delta,
        "rank_delta": rank_delta,
        "recent_form_delta": a_form - b_form,
        "goal_diff_recent_delta": a_gd - b_gd,
        "rest_days_delta": rest_delta,
        LIVE_ONLY_FEATURE: wc_gd_delta,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(root))
    from src.clean import load_and_clean_shootouts

    processed = root / "data" / "processed"
    matches = pd.read_csv(processed / "processed_matches.csv", parse_dates=["date"])
    rankings = pd.read_csv(processed / "processed_rankings.csv", parse_dates=["date"])
    shootouts = load_and_clean_shootouts(root / "data" / "raw" / "shootouts.csv")
    shootout_lookup = {(r.date, r.home_team, r.away_team): r.winner for r in shootouts.itertuples(index=False)}

    feat = build_training_features(matches, rankings, shootout_lookup)
    feat.to_csv(processed / "training_features.csv", index=False)
    print(f"Built {len(feat)} training rows with target distribution:\n{feat['target'].value_counts()}")
