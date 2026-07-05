# Landsat-9 Dashboard

Deployable dashboard for the notebook-aligned Landsat-9 TIR final pipeline:

```text
Raw TIR 200m 256x256 -> SimpleSRNet -> SR TIR 100m 512x512
  -> four 256x256 colorization tiles -> RGB-like 512x512 output
```

The backend calls the real repo functions:

```text
predict_sr_from_raw_array
colorize_512_tir_by_tiles
save_final_outputs
```

Preview PNGs are display-only and are never used as model input.

## Stack

```text
Frontend: static HTML/CSS/JavaScript served by FastAPI
Backend: FastAPI
Model runtime: PyTorch inference code from src.notebook_pipeline
```

This keeps deployment simple while still providing a real API-backed dashboard.

## Local Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r apps\dashboard\backend\requirements.txt
```

Set runtime paths:

```powershell
$env:SAVE_DIR=".\checkpoints"
$env:PREPROCESS_STATS_PATH=".\checkpoints\preprocess_stats.json"
$env:OUTPUT_DIR=".\outputs\dashboard_runs"
$env:COLOR_MODEL_DEFAULT="cnn"
$env:MAX_UPLOAD_MB="100"
$env:DASHBOARD_DEMO_MODE="false"
```

Start the backend and frontend together:

```powershell
uvicorn apps.dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The static frontend is served by FastAPI, so there is no separate frontend build step.

## Required Runtime Files

Place these under `SAVE_DIR` unless you override the environment variables:

```text
preprocess_stats.json
sr_cnn_residual_original_sensor_values.pth
color_cnn_unet_original_sensor_values.pth
color_pix2pix_original_sensor_values.pth        # only for gan mode
color_tiny_transformer_original_sensor_values.pth # only for transformer mode
```

Default production inference uses:

```text
sr_cnn_residual_original_sensor_values.pth
color_cnn_unet_original_sensor_values.pth
preprocess_stats.json
```

## Demo Mode

For UI-only testing without PyTorch or checkpoints:

```powershell
$env:DASHBOARD_DEMO_MODE="true"
uvicorn apps.dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

Demo mode is clearly marked in `/health`, API responses, and dashboard logs. Production mode remains the real inference path and returns clear errors when required stats or checkpoints are missing.

## API

```text
GET  /health
POST /api/inspect
POST /api/infer
GET  /api/download/{run_id}/{filename}
```

The download endpoint only serves files created under `OUTPUT_DIR`.

## Docker Deployment

Build from the repository root:

```powershell
docker build -f apps/dashboard/backend/Dockerfile -t landsat9-dashboard .
```

Run with mounted checkpoints and output storage:

```powershell
docker run --rm -p 8000:8000 `
  -e SAVE_DIR=/app/checkpoints `
  -e PREPROCESS_STATS_PATH=/app/checkpoints/preprocess_stats.json `
  -e OUTPUT_DIR=/app/outputs/dashboard_runs `
  -e COLOR_MODEL_DEFAULT=cnn `
  -e DASHBOARD_DEMO_MODE=false `
  -v ${PWD}\checkpoints:/app/checkpoints `
  -v ${PWD}\outputs:/app/outputs `
  landsat9-dashboard
```

Deploy the container on Render, Railway, Hugging Face Spaces Docker, or a VPS. Keep checkpoints and generated outputs in mounted/private storage, not in Git.

## Frontend Deployment

The frontend is served from `apps/dashboard/frontend` by the FastAPI service. For a separate static host, set `window.DASHBOARD_API_BASE` before loading `app.js` so the static page points to the deployed backend.

No committed files should include uploaded arrays, generated runs, checkpoints, `.env`, `.gh`, or secrets.
