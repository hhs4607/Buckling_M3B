from pydantic import BaseModel, Field, field_validator
from typing import Literal


class BeamParams(BaseModel):
    """Beam parameters matching desktop GUI defaults and validation rules."""

    # Geometry
    L: float = Field(1.5, gt=0, description="Beam length from root to tip (m)")
    b_root: float = Field(0.08, gt=0, description="Beam width at root (m)")
    b_tip: float = Field(0.04, gt=0, description="Beam width at tip (m)")
    h_root: float = Field(0.025, gt=0, description="Core height at root (m)")
    h_tip: float = Field(0.015, gt=0, description="Core height at tip (m)")
    w_f: float = Field(0.02, gt=0, description="Flange width (m)")

    # Face laminate
    t_face_total: float = Field(0.002, gt=0, description="Total face laminate thickness (m)")
    face_angles: str = Field("0,45,-45,90", description="Face ply angles (comma-separated degrees)")

    # Web laminate
    t_web_total: float = Field(0.0015, gt=0, description="Total web laminate thickness (m)")
    web_angles: str = Field("0,90", description="Web ply angles (comma-separated degrees)")

    # Material properties
    Ef: float = Field(230e9, gt=0, description="Fiber elastic modulus (Pa)")
    Em: float = Field(3.5e9, gt=0, description="Matrix elastic modulus (Pa)")
    Gf: float = Field(90e9, gt=0, description="Fiber shear modulus (Pa)")
    nuf: float = Field(0.2, gt=-1, lt=0.5, description="Fiber Poisson ratio")
    num: float = Field(0.35, gt=-1, lt=0.5, description="Matrix Poisson ratio")
    Vf: float = Field(0.6, gt=0, lt=1, description="Fiber volume fraction")

    # Boundary conditions
    Ktheta_root_per_m: float = Field(1e9, ge=0, description="Root spring stiffness (N*m/m)")

    # Solver settings
    PPW: int = Field(60, ge=10, description="Points per wavelength")
    nx_min: int = Field(1801, ge=10, description="Minimum grid points along beam")

    @field_validator("face_angles", "web_angles")
    @classmethod
    def validate_ply_angles(cls, v: str) -> str:
        parts = v.replace(" ", "").split(",")
        if not parts or parts == [""]:
            raise ValueError("Ply angles must be comma-separated numbers")
        for part in parts:
            try:
                float(part)
            except ValueError:
                raise ValueError(f"Invalid angle value: '{part}'")
        return v


class BucklingRequest(BaseModel):
    """Request body for POST /api/buckling/run."""

    core: Literal["m2", "m3"] = Field("m3", description="Solver core: m2 (fast) or m3 (accurate)")
    params: BeamParams = Field(default_factory=BeamParams)


class CurveData(BaseModel):
    """Load-deflection curve data arrays."""

    P: list[float]
    delta_linear: list[float]
    delta_nonlinear: list[float]
    delta_total: list[float]


class ContourData(BaseModel):
    """Mode shape contour grid data."""

    x: list[float]
    y: list[list[float]]
    w_normalized: list[list[float]]
    nx: int
    ny: int


class BucklingResponse(BaseModel):
    """Response body for POST /api/buckling/run."""

    core: str
    Pcr: float
    dcr: float
    alpha_star: float
    beta_star: float
    lambda_x: float
    curves: CurveData
    contour: ContourData
