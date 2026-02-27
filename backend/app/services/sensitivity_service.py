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
from app.schemas.sensitivity import SweepParam

_executor = ThreadPoolExecutor(max_workers=4)

# In-memory job store
_jobs: dict[str, dict] = {}


def _params_to_dict(params: BeamParams) -> dict:
    return params.model_dump()


def _eval_pcr(vals: dict, core: str) -> float:
    """Evaluate Pcr for a given parameter set."""
    if core == "m3":
        return float(eval_m3_Pcr_and_mode(vals, return_mode=False))
    else:
        return float(eval_m2_Pcr(vals))


def _run_sensitivity_sync(
    job_id: str,
    core: str,
    baseline_dict: dict,
    sweep_params: list[SweepParam],
):
    """Run sensitivity analysis synchronously, storing progress in job store."""
    total = sum(sp.points for sp in sweep_params)
    _jobs[job_id]["total"] = total
    _jobs[job_id]["status"] = "running"

    count = 0
    all_results = []

    # Compute baseline Pcr
    baseline_pcr = _eval_pcr(baseline_dict, core)

    for sp in sweep_params:
        v0 = float(baseline_dict[sp.key])
        pct = sp.percent / 100.0
        grid = np.linspace(v0 * (1 - pct), v0 * (1 + pct), sp.points)
        values = []
        pcr_values = []

        for val in grid:
            v = baseline_dict.copy()
            v[sp.key] = float(val)
            pcr = _eval_pcr(v, core)
            values.append(float(val))
            pcr_values.append(pcr)
            count += 1
            _jobs[job_id]["progress"] = {
                "current": count,
                "total": total,
                "param": sp.key,
                "message": f"Evaluating {sp.key} ({count}/{total})",
            }

        all_results.append({
            "param": sp.key,
            "values": values,
            "pcr_values": pcr_values,
        })

    _jobs[job_id]["result"] = {
        "results": all_results,
        "baseline_pcr": baseline_pcr,
    }
    _jobs[job_id]["status"] = "done"


def start_sensitivity_job(
    core: str,
    baseline_params: BeamParams,
    sweep_params: list[SweepParam],
) -> tuple[str, int]:
    """Start a sensitivity job in background thread. Returns (job_id, total_evaluations)."""
    job_id = f"sens_{uuid.uuid4().hex[:12]}"
    total = sum(sp.points for sp in sweep_params)
    baseline_dict = _params_to_dict(baseline_params)

    _jobs[job_id] = {
        "status": "pending",
        "progress": None,
        "result": None,
        "total": total,
    }

    _executor.submit(_run_sensitivity_sync, job_id, core, baseline_dict, sweep_params)
    return job_id, total


async def stream_sensitivity(job_id: str) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE events for a sensitivity job."""
    if job_id not in _jobs:
        yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
        return

    last_progress = 0
    while True:
        job = _jobs[job_id]

        # Stream progress
        if job["progress"] and job["progress"]["current"] > last_progress:
            last_progress = job["progress"]["current"]
            yield f"event: progress\ndata: {json.dumps(job['progress'])}\n\n"

        # Stream result when done
        if job["status"] == "done":
            yield f"event: result\ndata: {json.dumps(job['result'])}\n\n"
            yield "event: done\ndata: {}\n\n"
            # Cleanup
            del _jobs[job_id]
            return

        if job["status"] == "error":
            yield f"event: error\ndata: {json.dumps({'message': job.get('error', 'Unknown error')})}\n\n"
            del _jobs[job_id]
            return

        await asyncio.sleep(0.1)
