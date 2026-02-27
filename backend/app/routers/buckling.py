from fastapi import APIRouter, HTTPException

from app.schemas.buckling import BucklingRequest, BucklingResponse
from app.services.buckling_service import run_buckling_analysis

router = APIRouter(prefix="/api/buckling", tags=["buckling"])


@router.post("/run", response_model=BucklingResponse)
async def buckling_run(request: BucklingRequest) -> BucklingResponse:
    """Run single-case buckling analysis (M2 or M3 solver)."""
    try:
        result = await run_buckling_analysis(request.core, request.params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
