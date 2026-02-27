from pydantic import BaseModel, Field, field_validator
from typing import Literal

from app.schemas.buckling import BeamParams


class UncertainParam(BaseModel):
    """Configuration for one uncertain parameter."""

    key: str = Field(..., description="Parameter key (e.g., 'L', 'b_root')")
    low: float = Field(..., description="Lower bound")
    high: float = Field(..., description="Upper bound")

    @field_validator("high")
    @classmethod
    def high_must_exceed_low(cls, v: float, info) -> float:
        low = info.data.get("low")
        if low is not None and v <= low:
            raise ValueError(f"high ({v}) must be greater than low ({low})")
        return v


class SobolRequest(BaseModel):
    """Request body for POST /api/sobol/run."""

    core: Literal["m2", "m3"] = Field("m2", description="Solver core")
    baseline_params: BeamParams = Field(default_factory=BeamParams)
    uncertain_params: list[UncertainParam] = Field(..., min_length=1)
    n_base: int = Field(100, ge=10, le=10000, description="Saltelli base sample size")
    seed: int = Field(1234, ge=0, description="Random seed for reproducibility")


class SobolJobResponse(BaseModel):
    """Immediate response from POST /api/sobol/run."""

    job_id: str
    total_evaluations: int
    stream_url: str


class SobolProgressEvent(BaseModel):
    """SSE progress event data."""

    current: int
    total: int
    phase: str
    message: str


class SobolResultEvent(BaseModel):
    """SSE result event data."""

    names: list[str]
    S1: list[float]
    ST: list[float]
