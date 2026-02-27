from fastapi import APIRouter
from pydantic import BaseModel, ValidationError

from app.schemas.buckling import BeamParams

router = APIRouter(prefix="/api/config", tags=["config"])


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[dict[str, str]]


@router.post("/validate", response_model=ValidateResponse)
async def validate_config(params: dict) -> ValidateResponse:
    """Validate a set of beam parameters against all validation rules."""
    try:
        BeamParams(**params)
        return ValidateResponse(valid=True, errors=[])
    except ValidationError as e:
        errors = []
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            errors.append({"field": field, "message": err["msg"]})
        return ValidateResponse(valid=False, errors=errors)
