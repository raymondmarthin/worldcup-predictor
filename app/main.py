"""
main.py - FastAPI service for the World Cup 2026 Live Knockout Predictor.

Endpoints:
  GET /health        -> liveness check
  GET /model-info     -> model version, training date, features, backtest metrics
  GET /predict         -> win probability for team_a vs team_b
  GET /bracket          -> predictions for all remaining scheduled WC2026 matches (bonus)
"""
from __future__ import annotations
import sys
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import KnockoutPredictor
from app.schemas import HealthResponse, ModelInfoResponse, PredictResponse

predictor: KnockoutPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = KnockoutPredictor()
    yield


app = FastAPI(
    title="World Cup 2026 Live Knockout Predictor API",
    description="Predicts knockout-stage win probabilities using historical "
                "match data, FIFA ranking, and live World Cup 2026 context.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(**predictor.metadata)


@app.get("/predict", response_model=PredictResponse)
def predict(
    team_a: str = Query(..., description="First team name, e.g. France"),
    team_b: str = Query(..., description="Second team name, e.g. Brazil"),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if team_a.strip().lower() == team_b.strip().lower():
        raise HTTPException(status_code=400, detail="team_a and team_b must be different")
    if not predictor.known_team(team_a):
        raise HTTPException(status_code=404, detail=f"Unknown team: {team_a}")
    if not predictor.known_team(team_b):
        raise HTTPException(status_code=404, detail=f"Unknown team: {team_b}")

    result = predictor.predict(team_a, team_b)
    return PredictResponse(**result)


@app.get("/bracket")
def bracket():
    """Bonus endpoint: predicts every remaining scheduled WC2026 match."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    upcoming = predictor.wc2026[predictor.wc2026["status"] != "Completed"]
    predictions = []
    for row in upcoming.itertuples(index=False):
        try:
            pred = predictor.predict(row.home_team_name, row.away_team_name)
            pred["stage"] = row.stage_name
            pred["date"] = str(row.date.date())
            predictions.append(pred)
        except Exception:
            continue
    return {"count": len(predictions), "predictions": predictions}
