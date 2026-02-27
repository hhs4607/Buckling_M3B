import numpy as np
from math import pi
from concurrent.futures import ThreadPoolExecutor

from app.engine.buckling_analysis_M3a import (
    eval_m2_Pcr_with_mode,
    eval_m3_Pcr_and_mode,
    koiter_curves_from_mode,
)
from app.schemas.buckling import (
    BeamParams,
    BucklingResponse,
    CurveData,
    ContourData,
)

_executor = ThreadPoolExecutor(max_workers=4)


def _params_to_dict(params: BeamParams) -> dict:
    """Convert Pydantic BeamParams to the dict format expected by the engine."""
    return {
        "L": params.L,
        "b_root": params.b_root,
        "b_tip": params.b_tip,
        "h_root": params.h_root,
        "h_tip": params.h_tip,
        "w_f": params.w_f,
        "t_face_total": params.t_face_total,
        "face_angles": params.face_angles,
        "t_web_total": params.t_web_total,
        "web_angles": params.web_angles,
        "Ef": params.Ef,
        "Em": params.Em,
        "Gf": params.Gf,
        "nuf": params.nuf,
        "num": params.num,
        "Vf": params.Vf,
        "Ktheta_root_per_m": params.Ktheta_root_per_m,
        "PPW": params.PPW,
        "nx_min": params.nx_min,
    }


def _generate_contour(mode: dict, nx: int = 80, ny: int = 40) -> ContourData:
    """Generate mode contour grid data from mode dict (same logic as plot_contour_on_ax)."""
    x = mode["x"]
    b = mode["b"]
    F = mode["F"]

    X_lin = np.linspace(x[0], x[-1], nx)
    bX = np.interp(X_lin, x, b)
    FX = np.interp(X_lin, x, F)

    Y2d = np.zeros((nx, ny))
    W = np.zeros((nx, ny))

    for i in range(nx):
        y_phys = np.linspace(0.0, bX[i], ny)
        y_plot = y_phys - bX[i] / 2.0
        Y2d[i, :] = y_plot
        ky_i = pi / max(bX[i], 1e-18)
        W[i, :] = float(FX[i]) * np.sin(ky_i * y_phys)

    maxabs = max(np.max(np.abs(W)), 1e-18)
    Wn = W / maxabs

    return ContourData(
        x=X_lin.tolist(),
        y=Y2d.tolist(),
        w_normalized=Wn.tolist(),
        nx=nx,
        ny=ny,
    )


def _run_analysis(core: str, vals: dict) -> BucklingResponse:
    """Run buckling analysis synchronously (called in thread pool)."""
    PPW = int(vals.get("PPW", 60))
    nx_min = int(vals.get("nx_min", 1801))

    if core == "m3":
        mode = eval_m3_Pcr_and_mode(vals, PPW=PPW, nx_min=nx_min, return_mode=True)
    else:
        mode = eval_m2_Pcr_with_mode(vals, PPW=PPW, nx_min=nx_min, return_mode=True)

    P, dlin, dloc, dtot, dcr, _ = koiter_curves_from_mode(mode)

    curves = CurveData(
        P=P.tolist(),
        delta_linear=dlin.tolist(),
        delta_nonlinear=dloc.tolist(),
        delta_total=dtot.tolist(),
    )

    contour = _generate_contour(mode)

    return BucklingResponse(
        core=core.upper(),
        Pcr=float(mode["Pcr"]),
        dcr=float(dcr),
        alpha_star=float(mode["alpha_star"]),
        beta_star=float(mode["beta_star"]),
        lambda_x=float(mode["lambda_x"]),
        curves=curves,
        contour=contour,
    )


async def run_buckling_analysis(core: str, params: BeamParams) -> BucklingResponse:
    """Run buckling analysis in thread pool (non-blocking for async event loop)."""
    import asyncio

    vals = _params_to_dict(params)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_analysis, core, vals)
