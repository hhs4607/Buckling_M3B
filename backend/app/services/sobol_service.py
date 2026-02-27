import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

import numpy as np

from app.engine.buckling_analysis_M3a import (
    eval_m2_Pcr,
    eval_m3_Pcr_and_mode,
)
from app.schemas.buckling import BeamParams
from app.schemas.sobol import UncertainParam

_executor = ThreadPoolExecutor(max_workers=4)

# In-memory job store
_jobs: dict[str, dict] = {}


def _params_to_dict(params: BeamParams) -> dict:
    return params.model_dump()


def _eval_pcr(vals: dict, core: str) -> float:
    if core == "m3":
        return float(eval_m3_Pcr_and_mode(vals, return_mode=False))
    else:
        return float(eval_m2_Pcr(vals))


def _run_sobol_sync(
    job_id: str,
    core: str,
    baseline_dict: dict,
    uncertain_params: list[UncertainParam],
    n_base: int,
    seed: int,
):
    """Run Sobol analysis synchronously (Saltelli-lite), storing progress in job store."""
    names = [up.key for up in uncertain_params]
    k = len(names)
    total_evals = n_base * (2 + k)

    _jobs[job_id]["total"] = total_evals
    _jobs[job_id]["status"] = "running"

    rng = np.random.default_rng(seed)

    lows = np.array([up.low for up in uncertain_params])
    highs = np.array([up.high for up in uncertain_params])

    # Generate Saltelli matrices
    A = lows + (highs - lows) * rng.random((n_base, k))
    B = lows + (highs - lows) * rng.random((n_base, k))

    count = 0

    def eval_row(row):
        v = baseline_dict.copy()
        for j, name in enumerate(names):
            v[name] = float(row[j])
        return _eval_pcr(v, core)

    # Evaluate matrix A
    YA = np.zeros(n_base)
    for n in range(n_base):
        YA[n] = eval_row(A[n, :])
        count += 1
        _jobs[job_id]["progress"] = {
            "current": count,
            "total": total_evals,
            "phase": "Matrix A",
            "message": f"Matrix A ({n + 1}/{n_base})",
        }

    # Evaluate matrix B
    YB = np.zeros(n_base)
    for n in range(n_base):
        YB[n] = eval_row(B[n, :])
        count += 1
        _jobs[job_id]["progress"] = {
            "current": count,
            "total": total_evals,
            "phase": "Matrix B",
            "message": f"Matrix B ({n + 1}/{n_base})",
        }

    # Evaluate AB matrices
    YAB_list = []
    for i in range(k):
        M = A.copy()
        M[:, i] = B[:, i]
        YAB = np.zeros(n_base)
        for n in range(n_base):
            YAB[n] = eval_row(M[n, :])
            count += 1
            _jobs[job_id]["progress"] = {
                "current": count,
                "total": total_evals,
                "phase": f"AB[{names[i]}]",
                "message": f"AB matrix for {names[i]} ({n + 1}/{n_base})",
            }
        YAB_list.append(YAB)

    # Compute Sobol indices
    Y_all = np.concatenate([YA, YB])
    V = np.var(Y_all, ddof=1) if len(Y_all) > 1 else 1.0

    S1 = np.zeros(k)
    ST = np.zeros(k)
    for i in range(k):
        YAB = YAB_list[i]
        S1[i] = np.mean(YB * (YAB - YA)) / V if V > 0 else 0.0
        ST[i] = 0.5 * np.mean((YA - YAB) ** 2) / V if V > 0 else 0.0

    # Sort by ST descending
    order = np.argsort(-ST)
    names_sorted = [names[i] for i in order]
    S1_sorted = S1[order].tolist()
    ST_sorted = ST[order].tolist()

    _jobs[job_id]["result"] = {
        "names": names_sorted,
        "S1": S1_sorted,
        "ST": ST_sorted,
    }
    _jobs[job_id]["status"] = "done"


def start_sobol_job(
    core: str,
    baseline_params: BeamParams,
    uncertain_params: list[UncertainParam],
    n_base: int,
    seed: int,
) -> tuple[str, int]:
    """Start a Sobol job in background thread. Returns (job_id, total_evaluations)."""
    job_id = f"sobol_{uuid.uuid4().hex[:12]}"
    k = len(uncertain_params)
    total = n_base * (2 + k)
    baseline_dict = _params_to_dict(baseline_params)

    _jobs[job_id] = {
        "status": "pending",
        "progress": None,
        "result": None,
        "total": total,
    }

    _executor.submit(
        _run_sobol_sync, job_id, core, baseline_dict, uncertain_params, n_base, seed
    )
    return job_id, total


async def stream_sobol(job_id: str) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE events for a Sobol job."""
    if job_id not in _jobs:
        yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
        return

    last_progress = 0
    while True:
        job = _jobs[job_id]

        if job["progress"] and job["progress"]["current"] > last_progress:
            last_progress = job["progress"]["current"]
            yield f"event: progress\ndata: {json.dumps(job['progress'])}\n\n"

        if job["status"] == "done":
            yield f"event: result\ndata: {json.dumps(job['result'])}\n\n"
            yield "event: done\ndata: {}\n\n"
            del _jobs[job_id]
            return

        if job["status"] == "error":
            yield f"event: error\ndata: {json.dumps({'message': job.get('error', 'Unknown error')})}\n\n"
            del _jobs[job_id]
            return

        await asyncio.sleep(0.1)
