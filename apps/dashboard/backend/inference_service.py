"""Inference service for the deployable Landsat-9 dashboard.

Production inference uses the notebook-aligned path from ``src.notebook_pipeline``:
raw TIR -> SimpleSRNet -> tile-wise colorization -> saved numerical outputs.
Preview images are created only for display and are never used as model input.
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
import time
import uuid
import importlib.util
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_FILENAMES = {
    "sr": "sr_cnn_residual_original_sensor_values.pth",
    "cnn": "color_cnn_unet_original_sensor_values.pth",
    "gan": "color_pix2pix_original_sensor_values.pth",
    "transformer": "color_tiny_transformer_original_sensor_values.pth",
}

VALID_COLOR_CHOICES = {"cnn", "gan", "transformer"}


class DashboardError(Exception):
    def __init__(self, message: str, status_code: int = 400, logs: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.logs = logs or []


@dataclass(frozen=True)
class DashboardSettings:
    save_dir: Path
    preprocess_stats_path: Path
    output_dir: Path
    default_color_model: str
    max_upload_mb: int
    demo_mode: bool

    @classmethod
    def from_env(cls) -> "DashboardSettings":
        save_dir = Path(os.getenv("SAVE_DIR", REPO_ROOT / "checkpoints")).resolve()
        stats_path = Path(
            os.getenv("PREPROCESS_STATS_PATH", save_dir / "preprocess_stats.json")
        ).resolve()
        output_dir = Path(
            os.getenv("OUTPUT_DIR", REPO_ROOT / "outputs" / "dashboard_runs")
        ).resolve()
        color_default = os.getenv("COLOR_MODEL_DEFAULT", "cnn").strip().lower()
        if color_default not in VALID_COLOR_CHOICES:
            color_default = "cnn"
        demo = os.getenv("DASHBOARD_DEMO_MODE", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "100"))
        return cls(
            save_dir=save_dir,
            preprocess_stats_path=stats_path,
            output_dir=output_dir,
            default_color_model=color_default,
            max_upload_mb=max_upload_mb,
            demo_mode=demo,
        )


class InferenceService:
    def __init__(self, settings: DashboardSettings | None = None):
        self.settings = settings or DashboardSettings.from_env()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._sr_model = None
        self._color_models: dict[str, Any] = {}

    def health(self) -> dict[str, Any]:
        readiness = self.model_readiness(self.settings.default_color_model)
        models_loaded = self.settings.demo_mode or (
            self._sr_model is not None
            and self.settings.default_color_model in self._color_models
        )
        return {
            "status": "ok",
            "models_loaded": models_loaded,
            "default_color_model": self.settings.default_color_model,
            "demo_mode": self.settings.demo_mode,
            "model_ready": readiness["ready"],
            "missing": readiness["missing"],
        }

    def model_readiness(self, color_choice: str = "cnn") -> dict[str, Any]:
        missing: list[str] = []
        if self.settings.demo_mode:
            return {"ready": True, "missing": missing}
        if importlib.util.find_spec("torch") is None:
            missing.append("Python package `torch`")
        if not self.settings.preprocess_stats_path.is_file():
            missing.append(str(self.settings.preprocess_stats_path))
        for key in ["sr", color_choice]:
            checkpoint = self.settings.save_dir / CHECKPOINT_FILENAMES[key]
            if not checkpoint.is_file():
                missing.append(str(checkpoint))
        return {"ready": not missing, "missing": missing}

    def inspect_upload(self, filename: str, content: bytes) -> dict[str, Any]:
        filename = Path(filename or "").name
        if Path(filename).suffix.lower() != ".npy":
            return self._invalid_summary(
                filename,
                "Upload a raw/original sensor-value TIR .npy file. Preview images are rejected.",
            )
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            return self._invalid_summary(
                filename,
                f"File exceeds MAX_UPLOAD_MB={self.settings.max_upload_mb}.",
            )
        try:
            arr = np.load(BytesIO(content), allow_pickle=False)
        except Exception as exc:
            return self._invalid_summary(filename, f"Could not read .npy file: {exc}")
        return self._summarize_array(filename, arr)

    def run_inference(
        self,
        filename: str,
        content: bytes,
        color_choice: str,
        save_preview: bool = True,
        include_npy: bool = True,
    ) -> dict[str, Any]:
        color_choice = (color_choice or self.settings.default_color_model).strip().lower()
        if color_choice not in VALID_COLOR_CHOICES:
            raise DashboardError("Color model must be one of: cnn, gan, transformer.")

        logs = ["File uploaded"]
        summary = self.inspect_upload(filename, content)
        if not summary["valid"]:
            raise DashboardError(summary["message"], logs=logs)
        logs.append("Shape validated: raw 200m TIR array is 256x256")

        arr = np.load(BytesIO(content), allow_pickle=False).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        run_id = self._new_run_id()
        run_dir = self.settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        stem = self._safe_stem(filename)
        prefix = run_dir / f"{stem}_{color_choice}"

        if self.settings.demo_mode:
            logs.append("Demo mode enabled: generating UI-only synthetic outputs")
            pred_tir, pred_rgb = self._run_demo_inference(arr)
            saved = self._save_demo_outputs(prefix, pred_tir, pred_rgb)
        else:
            readiness = self.model_readiness(color_choice)
            if not readiness["ready"]:
                missing = "\n".join(readiness["missing"])
                raise DashboardError(
                    "Production inference needs preprocessing stats and checkpoints:\n" + missing,
                    status_code=503,
                    logs=logs,
                )
            try:
                sr_model, color_model, stats = self._load_real_models(color_choice)
                logs.append("SR model loaded")
                logs.append("Color model loaded")
                pred_tir = self._predict_sr(sr_model, arr, stats)
                logs.append("SR inference complete")
                pred_rgb = self._colorize_tiles(color_model, pred_tir, stats)
                logs.append("Tile-wise colorization complete")
            except ModuleNotFoundError as exc:
                raise DashboardError(
                    f"Production inference dependency is missing: {exc.name}.",
                    status_code=503,
                    logs=logs,
                ) from exc
            except Exception as exc:
                raise DashboardError(
                    f"Production inference failed: {exc}",
                    status_code=500,
                    logs=logs,
                ) from exc
            saved = self._save_final_outputs(prefix, pred_tir, pred_rgb)

        logs.append("Outputs saved")

        raw_preview = run_dir / f"{stem}_raw_tir200m_preview.png"
        sr_preview = run_dir / f"{stem}_{color_choice}_sr_tir100m_preview.png"
        if save_preview:
            self._save_tir_preview(arr, raw_preview, "Raw TIR 200m preview only")
            self._save_tir_preview(pred_tir, sr_preview, "SR TIR 100m preview only")

        manifest_path = run_dir / "inference_manifest.json"
        outputs = self._build_outputs(run_id, saved, raw_preview, sr_preview, manifest_path)
        metrics = {
            "input_shape": list(arr.shape),
            "input_range": [float(np.min(arr)), float(np.max(arr))],
            "sr_shape": list(pred_tir.shape),
            "sr_range": [float(np.min(pred_tir)), float(np.max(pred_tir))],
            "rgb_shape": list(pred_rgb.shape),
            "rgb_range": [float(np.min(pred_rgb)), float(np.max(pred_rgb))],
            "color_choice": color_choice,
            "include_npy": include_npy,
            "save_preview": save_preview,
        }
        manifest = {
            "run_id": run_id,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "demo_mode": self.settings.demo_mode,
            "input_filename": Path(filename).name,
            "input_summary": summary,
            "pipeline": [
                "raw TIR 200m 256x256",
                "SimpleSRNet super-resolution",
                "four tile ColorUNet/GAN/Transformer colorization",
                "merged RGB-like 512x512 output",
            ],
            "preview_note": "Preview PNGs are display-only and are never used as model input.",
            "outputs": {key: value for key, value in outputs.items()},
            "metrics_or_stats": metrics,
            "logs": logs,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        outputs = self._build_outputs(run_id, saved, raw_preview, sr_preview, manifest_path)

        return {
            "success": True,
            "run_id": run_id,
            "demo_mode": self.settings.demo_mode,
            "input_summary": summary,
            "outputs": outputs,
            "metrics_or_stats": metrics,
            "logs": logs,
        }

    def resolve_download(self, run_id: str, filename: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id or ""):
            raise DashboardError("Invalid run id.")
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise DashboardError("Invalid filename.")
        root = self.settings.output_dir.resolve()
        run_dir = (root / run_id).resolve()
        path = (run_dir / filename).resolve()
        if os.path.commonpath([str(root), str(path)]) != str(root):
            raise DashboardError("Download path is outside the output directory.")
        if not path.is_file():
            raise DashboardError("Requested output file was not found.", status_code=404)
        return path

    def _load_real_models(self, color_choice: str):
        from src.models.simple_sr import SimpleSRNet
        from src.notebook_pipeline import load_best_color_model, load_model_checkpoint
        from src.preprocessing import load_preprocess_stats

        stats = load_preprocess_stats(str(self.settings.preprocess_stats_path), strict=True)
        if self._sr_model is None:
            self._sr_model = load_model_checkpoint(
                SimpleSRNet(channels=64, num_blocks=6),
                str(self.settings.save_dir / CHECKPOINT_FILENAMES["sr"]),
            )
        if color_choice not in self._color_models:
            self._color_models[color_choice] = load_best_color_model(
                color_choice,
                save_dir=str(self.settings.save_dir),
            )
        return self._sr_model, self._color_models[color_choice], stats

    @staticmethod
    def _predict_sr(sr_model, arr: np.ndarray, stats):
        from src.notebook_pipeline import predict_sr_from_raw_array

        return predict_sr_from_raw_array(sr_model, arr, stats=stats)

    @staticmethod
    def _colorize_tiles(color_model, pred_tir: np.ndarray, stats):
        from src.notebook_pipeline import colorize_512_tir_by_tiles

        return colorize_512_tir_by_tiles(color_model, pred_tir, stats=stats)

    @staticmethod
    def _save_final_outputs(prefix: Path, pred_tir: np.ndarray, pred_rgb: np.ndarray):
        from src.notebook_pipeline import save_final_outputs

        return save_final_outputs(str(prefix), pred_tir, pred_rgb)

    def _save_demo_outputs(self, prefix: Path, pred_tir: np.ndarray, pred_rgb: np.ndarray):
        tir_path = Path(str(prefix) + "_pred_tir100m_512.npy")
        rgb_path = Path(str(prefix) + "_pred_rgb_chw_original_scale.npy")
        tif_path = Path(str(prefix) + "_pred_bgr_chw.tif")
        preview_path = Path(str(prefix) + "_preview.png")
        np.save(tir_path, pred_tir.astype(np.float32))
        np.save(rgb_path, pred_rgb.astype(np.float32))
        try:
            import tifffile

            tifffile.imwrite(tif_path, pred_rgb[[2, 1, 0], :, :].astype(np.float32))
        except Exception:
            tif_path = None
        self._save_rgb_preview(pred_rgb, preview_path)
        return {
            "tir_npy": str(tir_path),
            "rgb_npy": str(rgb_path),
            "tif": str(tif_path) if tif_path else None,
            "preview": str(preview_path),
        }

    @staticmethod
    def _run_demo_inference(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sr = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1).astype(np.float32)
        finite = sr[np.isfinite(sr)]
        lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
        norm = np.clip((sr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        rgb = np.stack(
            [
                norm,
                np.sqrt(np.clip(norm, 0.0, 1.0)),
                1.0 - np.clip(norm * 0.85, 0.0, 1.0),
            ],
            axis=0,
        ).astype(np.float32)
        return sr, rgb

    @staticmethod
    def _save_tir_preview(arr: np.ndarray, path: Path, title: str) -> None:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo, hi = np.percentile(finite, [2, 98])
            disp = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        else:
            disp = np.zeros_like(arr, dtype=np.float32)
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(5, 4), dpi=150)
            plt.imshow(disp, cmap="inferno")
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            fig.savefig(path)
            plt.close(fig)
        except ModuleNotFoundError:
            InferenceService._write_rgb_png(path, InferenceService._heatmap_display(disp))

    @staticmethod
    def _save_rgb_preview(rgb_chw: np.ndarray, path: Path) -> None:
        rgb_hwc = np.moveaxis(rgb_chw.astype(np.float32), 0, -1)
        finite = rgb_hwc[np.isfinite(rgb_hwc)]
        if finite.size:
            lo, hi = np.percentile(finite, [2, 98])
            disp = np.clip((rgb_hwc - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        else:
            disp = np.zeros_like(rgb_hwc, dtype=np.float32)
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(5, 5), dpi=150)
            plt.imshow(disp)
            plt.title("Preview only: stretched for display")
            plt.axis("off")
            plt.tight_layout()
            fig.savefig(path)
            plt.close(fig)
        except ModuleNotFoundError:
            InferenceService._write_rgb_png(path, disp)

    @staticmethod
    def _heatmap_display(gray: np.ndarray) -> np.ndarray:
        gray = np.clip(gray.astype(np.float32), 0.0, 1.0)
        red = np.clip(1.65 * gray - 0.18, 0.0, 1.0)
        green = np.clip(1.45 - np.abs(gray - 0.55) * 2.1, 0.0, 1.0)
        blue = np.clip(1.2 - 1.45 * gray, 0.0, 1.0)
        return np.stack([red, green, blue], axis=-1)

    @staticmethod
    def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
        rgb8 = np.clip(rgb, 0.0, 1.0)
        rgb8 = (rgb8 * 255.0 + 0.5).astype(np.uint8)
        if rgb8.ndim != 3 or rgb8.shape[2] != 3:
            raise ValueError(f"Expected RGB HWC image, got {rgb8.shape}")
        height, width, _channels = rgb8.shape

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        raw = b"".join(b"\x00" + rgb8[row].tobytes() for row in range(height))
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level=6))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png)

    @staticmethod
    def _safe_stem(filename: str) -> str:
        stem = Path(filename).stem or "sample"
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
        return stem[:80] or "sample"

    @staticmethod
    def _new_run_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

    @staticmethod
    def _invalid_summary(filename: str, message: str) -> dict[str, Any]:
        return {
            "valid": False,
            "filename": Path(filename or "").name,
            "shape": [],
            "dtype": "",
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "finite_count": 0,
            "message": message,
        }

    def _summarize_array(self, filename: str, arr: np.ndarray) -> dict[str, Any]:
        shape = [int(v) for v in arr.shape]
        if not np.issubdtype(arr.dtype, np.number):
            return self._invalid_summary(filename, "Expected a numeric .npy array.")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {
                "valid": False,
                "filename": filename,
                "shape": shape,
                "dtype": str(arr.dtype),
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "finite_count": 0,
                "message": "Array has no finite values.",
            }
        valid = tuple(arr.shape) == (256, 256)
        message = (
            "Valid raw TIR array"
            if valid
            else f"Expected input shape (256, 256), got {tuple(arr.shape)}. Upload raw 200m TIR .npy."
        )
        return {
            "valid": valid,
            "filename": filename,
            "shape": shape,
            "dtype": str(arr.dtype),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "finite_count": int(finite.size),
            "message": message,
        }

    def _file_record(self, run_id: str, path_value: str | Path | None, reason: str | None = None):
        if path_value is None:
            return {
                "filename": "",
                "url": None,
                "exists": False,
                "reason": reason or "Not generated",
            }
        path = Path(path_value)
        exists = path.is_file()
        filename = path.name
        return {
            "filename": filename,
            "url": f"/api/download/{quote(run_id)}/{quote(filename)}" if exists else None,
            "exists": exists,
            "reason": None if exists else reason or "Not generated",
        }

    def _build_outputs(
        self,
        run_id: str,
        saved: dict[str, Any],
        raw_preview: Path,
        sr_preview: Path,
        manifest_path: Path,
    ) -> dict[str, dict[str, Any]]:
        return {
            "raw_preview_png": self._file_record(run_id, raw_preview),
            "sr_preview_png": self._file_record(run_id, sr_preview),
            "tir100m_npy": self._file_record(run_id, saved.get("tir_npy")),
            "rgb_original_npy": self._file_record(run_id, saved.get("rgb_npy")),
            "bgr_tif": self._file_record(
                run_id,
                saved.get("tif"),
                reason="TIFF output requires tifffile at runtime.",
            ),
            "preview_png": self._file_record(run_id, saved.get("preview")),
            "manifest_json": self._file_record(run_id, manifest_path),
        }
