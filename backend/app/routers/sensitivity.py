from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.sensitivity import SensitivityRequest, SensJobResponse
from app.services.sensitivity_service import start_sensitivity_job, stream_sensitivity

router = APIRouter(prefix="/api/sensitivity", tags=["sensitivity"])


@router.post("/run", response_model=SensJobResponse)
async def sensitivity_run(request: SensitivityRequest) -> SensJobResponse:
    """Start OAT sensitivity analysis. Returns job_id for SSE streaming."""
    try:
        job_id, total = start_sensitivity_job(
            request.core, request.baseline_params, request.sweep_params
        )
        return SensJobResponse(
            job_id=job_id,
            total_evaluations=total,
            stream_url=f"/api/sensitivity/stream/{job_id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{job_id}")
async def sensitivity_stream(job_id: str):
    """SSE stream for sensitivity analysis progress and results."""
    return StreamingResponse(
        stream_sensitivity(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
