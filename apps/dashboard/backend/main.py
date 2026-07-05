"""FastAPI entrypoint for the Landsat-9 TIR dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from .inference_service import DashboardError, InferenceService
    from .schemas import HealthResponse, InferResponse, InspectResponse
except ImportError:  # Allows `uvicorn main:app` from this folder.
    from inference_service import DashboardError, InferenceService
    from schemas import HealthResponse, InferResponse, InspectResponse

app = FastAPI(
    title="Landsat-9 TIR Super-Resolution and Colorization Dashboard",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = InferenceService()


@app.exception_handler(DashboardError)
async def dashboard_error_handler(_request, exc: DashboardError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "logs": exc.logs},
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return service.health()


@app.post("/api/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)):
    content = await file.read()
    return service.inspect_upload(file.filename or "", content)


@app.post("/api/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    color_choice: str = Form("cnn"),
    save_preview: bool = Form(True),
    include_npy: bool = Form(True),
):
    content = await file.read()
    return service.run_inference(
        file.filename or "",
        content,
        color_choice=color_choice,
        save_preview=save_preview,
        include_npy=include_npy,
    )


@app.get("/api/download/{run_id}/{filename}")
async def download(run_id: str, filename: str):
    path = service.resolve_download(run_id, filename)
    return FileResponse(path, filename=path.name)


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
