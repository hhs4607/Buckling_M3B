from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.sobol import SobolRequest, SobolJobResponse
from app.services.sobol_service import start_sobol_job, stream_sobol

router = APIRouter(prefix="/api/sobol", tags=["sobol"])


@router.post("/run", response_model=SobolJobResponse)
async def sobol_run(request: SobolRequest) -> SobolJobResponse:
    """Start Sobol UQ analysis. Returns job_id for SSE streaming."""
    try:
        job_id, total = start_sobol_job(
            request.core,
            request.baseline_params,
            request.uncertain_params,
            request.n_base,
            request.seed,
        )
        return SobolJobResponse(
            job_id=job_id,
            total_evaluations=total,
            stream_url=f"/api/sobol/stream/{job_id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{job_id}")
async def sobol_stream(job_id: str):
    """SSE stream for Sobol analysis progress and results."""
    return StreamingResponse(
        stream_sobol(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
