from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    calibrated: bool
    trained_at: str
    split_date: str
    n_train: int
    n_test: int
    features_used: list[str]
    backtest_metrics: dict


class PredictResponse(BaseModel):
    team_a: str
    team_b: str
    win_a: float = Field(..., ge=0.0, le=1.0)
    win_b: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    confidence: str
    features_used: list[str]


class ErrorResponse(BaseModel):
    detail: str
