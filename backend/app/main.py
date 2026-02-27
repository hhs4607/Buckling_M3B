from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import buckling, sensitivity, sobol, config as config_router

app = FastAPI(
    title="E3B Buckling Analysis API",
    description="API for double-tapered composite box beam buckling analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app.include_router(buckling.router)
app.include_router(sensitivity.router)
app.include_router(sobol.router)
app.include_router(config_router.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
