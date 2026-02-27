from pydantic import BaseModel, Field, field_validator
from typing import Literal

from app.schemas.buckling import BeamParams


class SweepParam(BaseModel):
    """Configuration for one parameter to sweep."""

    key: str = Field(..., description="Parameter key (e.g., 'L', 'b_root')")
    percent: float = Field(10.0, gt=0, le=100, description="Sweep range as ± percentage")
    points: int = Field(5, ge=3, le=51, description="Number of evaluation points")


class SensitivityRequest(BaseModel):
    """Request body for POST /api/sensitivity/run."""

    core: Literal["m2", "m3"] = Field("m2", description="Solver core")
    baseline_params: BeamParams = Field(default_factory=BeamParams)
    sweep_params: list[SweepParam] = Field(..., min_length=1)

    @field_validator("sweep_params")
    @classmethod
    def validate_sweep_params(cls, v: list[SweepParam]) -> list[SweepParam]:
        if len(v) == 0:
            raise ValueError("At least one sweep parameter must be selected")
        return v


class SensJobResponse(BaseModel):
    """Immediate response from POST /api/sensitivity/run."""

    job_id: str
    total_evaluations: int
    stream_url: str


class SensProgressEvent(BaseModel):
    """SSE progress event data."""

    current: int
    total: int
    param: str
    message: str


class SensParamResult(BaseModel):
    """Result for one swept parameter."""

    param: str
    values: list[float]
    pcr_values: list[float]


class SensResultEvent(BaseModel):
    """SSE result event data."""

    results: list[SensParamResult]
    baseline_pcr: float
