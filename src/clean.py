"""
clean.py
Canonical team-name mapping + data cleaning.

Raw sources use inconsistent country names (e.g. "Cape Verde" vs "Cabo Verde",
"Turkey" vs "Türkiye"). This module maps everything to one canonical form so
matches, rankings, and 2026 live data can be joined reliably.

IMPORTANT: never edit data/raw/* manually. All cleaning happens here, in code.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

# Canonical name <- list of known aliases seen across the raw datasets.
# Canonical form chosen = the name used by martj42/international_results
# (our largest historical source), except where FIFA's official 2026 name
# is clearer (e.g. "Czechia").
CANONICAL_TEAM_MAP: dict[str, str] = {
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czech Republic": "Czechia",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Republic of Ireland": "Ireland",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "United States of America": "United States",
    "Saudi Arabia": "Saudi Arabia",
    "UAE": "United Arab Emirates",
}


def canonicalize(name: str) -> str:
    """Map a raw team name to its canonical form."""
    if not isinstance(name, str):
        return name
    name = name.strip()
    return CANONICAL_TEAM_MAP.get(name, name)


def load_and_clean_results(raw_path: Path) -> pd.DataFrame:
    """Load international_results.csv, apply canonical names, parse dates."""
    df = pd.read_csv(raw_path)
    df["home_team"] = df["home_team"].map(canonicalize)
    df["away_team"] = df["away_team"].map(canonicalize)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score", "date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def load_and_clean_shootouts(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["home_team"] = df["home_team"].map(canonicalize)
    df["away_team"] = df["away_team"].map(canonicalize)
    df["winner"] = df["winner"].map(canonicalize)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_and_clean_rankings(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["team"] = df["team"].map(canonicalize)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["team", "total_points", "date"])
    df = df.sort_values(["date", "total_points"], ascending=[True, False])
    # numeric rank within each ranking snapshot: 1 = strongest
    df["rank"] = df.groupby("date")["total_points"].rank(ascending=False, method="min")
    return df.sort_values(["team", "date"]).reset_index(drop=True)


def load_and_clean_wc2026(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["home_team_name"] = df["home_team_name"].map(canonicalize)
    df["away_team_name"] = df["away_team_name"].map(canonicalize)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def resolve_match_winner(row: pd.Series, shootout_lookup: dict) -> int | None:
    """
    Return 1 if home_team is the FINAL winner, 0 if away_team is, None if
    the match was a genuine draw with no decisive result (dropped from the
    binary knockout-winner training set).
    """
    if row["home_score"] > row["away_score"]:
        return 1
    if row["home_score"] < row["away_score"]:
        return 0
    # scores level -> check shootout record (extra-time/penalty decider)
    key = (row["date"], row["home_team"], row["away_team"])
    winner = shootout_lookup.get(key)
    if winner is None:
        return None
    if winner == row["home_team"]:
        return 1
    if winner == row["away_team"]:
        return 0
    return None


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    results = load_and_clean_results(raw / "international_results.csv")
    rankings = load_and_clean_rankings(raw / "fifa_ranking_historical.csv")
    wc2026 = load_and_clean_wc2026(raw / "wc2026_matches_detailed.csv")
    shootouts = load_and_clean_shootouts(raw / "shootouts.csv")

    processed = root / "data" / "processed"
    processed.mkdir(exist_ok=True)
    results.to_csv(processed / "processed_matches.csv", index=False)
    rankings.to_csv(processed / "processed_rankings.csv", index=False)
    wc2026.to_csv(processed / "processed_wc2026.csv", index=False)
    shootouts.to_csv(processed / "processed_shootouts.csv", index=False)
    print(f"results: {len(results)} rows, rankings: {len(rankings)} rows, "
          f"wc2026: {len(wc2026)} rows, shootouts: {len(shootouts)} rows")
