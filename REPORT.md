# REPORT — World Cup 2026 Live Knockout Predictor

## 1. Problem
Predict `win_a` / `win_b` probabilities for a World Cup 2026 knockout match
between two given teams. Knockout matches always have a final winner (extra
time / penalties), so draw is not modeled as an output class.

## 2. Data
- **Training/backtest**: 49,487 international matches (1872–2026,
  `martj42/international_results`), filtered to matches from **1993-01-01
  onward** (FIFA ranking coverage starts Dec 1992, and elo/rank features
  need a ranking snapshot before that date). After filtering to matches
  with a decisive winner and both teams having prior match history, the
  final training feature set has **20,779 rows**.
- **FIFA ranking proxy**: `Dato-Futbol/fifa-ranking`, Dec 1992–Sep 2024
  (335 snapshots, 235 teams). Used as `elo_delta`/`rank_delta` via
  as-of-date lookup (most recent snapshot strictly before the match date).
  Rankings after Sep 2024 are unavailable — the model falls back to the
  last known snapshot for recent/live matches (documented limitation).
- **Live 2026 context**: `mominullptr/FIFA-World-Cup-2026-Dataset`
  (`matches_detailed.csv`), 82 completed matches as of this report (through
  Round of 32, up to 2026-07-02). Used **only** to compute
  `world_cup_2026_goal_diff_delta` for `/predict` requests — never added
  to the training set.
- **Canonical team names**: raw sources disagree on names (e.g. "Cabo
  Verde" vs "Cape Verde", "Türkiye" vs "Turkey", "USA" vs "United States").
  A mapping table in `src/clean.py` normalizes all sources to one name
  before any join.

## 3. Feature engineering (all computed strictly before the match date)
| Feature | Definition | Status |
|---|---|---|
| `elo_delta` | FIFA ranking points, team_a − team_b, as-of match date | Wajib |
| `rank_delta` | FIFA rank position, team_a − team_b (lower = stronger) | Wajib |
| `recent_form_delta` | Avg points (3/1/0) over last 10 matches, team_a − team_b | Wajib |
| `goal_diff_recent_delta` | Avg goal difference over last 10 matches, team_a − team_b | Wajib |
| `world_cup_2026_goal_diff_delta` | Goal difference in completed WC2026 matches so far | Wajib, live-only (0 for historical training rows since 2026 wasn't in training) |
| `rest_days_delta` | Days since each team's last match, team_a − team_b | Opsional (included) |
| `h2h_delta` | Head-to-head record | Opsional (**not implemented** — see limitations) |
| `penalty_history_delta` | Penalty shootout proxy | Bonus (**not implemented**) |

Leakage guard: features are built with a single chronological pass per
team (rolling window updated *after* being read for the current match), and
covered by `tests/test_features.py::test_build_training_features_no_future_leakage`.

## 4. Model
- **Split**: temporal, train < 2018-01-01 (15,019 rows), test ≥ 2018-01-01
  (5,760 rows, covers WC 2018 + WC 2022 + recent friendlies/qualifiers).
- **Models compared**:

| Model | Log loss | Brier score | Accuracy |
|---|---|---|---|
| Logistic Regression (baseline) | **0.4783** | 0.1577 | 0.7667 |
| Random Forest | 0.4820 | 0.1586 | 0.7670 |
| XGBoost | 0.4803 | 0.1581 | 0.7651 |

- **Selected model**: Logistic Regression — lowest log loss on the held-out
  temporal test set, and simplest to explain/maintain. The three models are
  close enough that added complexity (RF/XGBoost) isn't clearly justified
  for this feature set.
- **Calibration**: Brier score (0.158) is below the 0.22 threshold used as
  an overconfidence heuristic in `src/train.py`, so `CalibratedClassifierCV`
  was **not** applied. The logic is in place and triggers automatically if
  a future retrain produces overconfident probabilities.
- **Artifact**: `models/model.joblib` bundles the fitted model + fitted
  `StandardScaler` in one object; `models/metadata.json` records version,
  training timestamp, split date, and all backtest metrics.

## 5. API behavior
- `/predict` always returns `win_a + win_b ≈ 1.0` (verified in
  `tests/test_api.py`).
- Unknown team names → `404`. Same team on both sides → `400`.
- `confidence` label is a simple margin heuristic (`|win_a − 0.5|`): low
  (<0.05), medium (<0.15), high (otherwise) — not a statistical confidence
  interval.
- `/bracket` (bonus) runs `/predict` over every non-completed WC2026 fixture
  currently in the live dataset.

## 6. Limitations
- **Not a betting-grade model.** Football has high outcome variance;
  accuracy (~77%) and log loss are backtest numbers on historical data, not
  a guarantee for future matches.
- **FIFA ranking data ends Sep 2024.** Team strength for 2025–2026 matches
  uses the last available snapshot, which slightly understates recent form
  changes — partially offset by `recent_form_delta` and the live
  `world_cup_2026_goal_diff_delta` feature.
- **h2h_delta and penalty_history_delta were not implemented** in this MVP
  (marked optional/bonus in the task spec); adding them is a natural next
  step if the intern has more time.
- **No explicit "knockout stage" label exists in the historical training
  data.** The model learns a general win/loss pattern from all
  international matches (friendlies + tournaments), not knockout matches
  specifically. This is a deliberate simplification: there's no reliable
  stage field across sources for historical matches, and the recommended
  binary target (decisive winner, draws resolved via shootout data)
  already captures the "someone must win" property required for knockout
  matches at inference time.
- **Team name coverage.** The canonical mapping table covers common
  naming mismatches for the 48 WC2026 teams; less common team-name
  variants elsewhere in the historical data are left as-is (they don't
  affect 2026 predictions, only some very old/regional-team training rows).

## 7. How to reproduce
```bash
python -m src.clean
python -m src.features
python -m src.train
pytest tests/ -v
uvicorn app.main:app --reload
```
