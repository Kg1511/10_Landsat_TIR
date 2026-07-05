"""Notebook-aligned final two-stage inference.

Default final model: SimpleSRNet + ColorUNet. GAN and transformer colorizers are
available as comparison/extension choices.
"""

import argparse
import os

from src.config import CHECKPOINT_DIR
from src.notebook_pipeline import (
    colorize_512_tir_by_tiles,
    load_best_color_model,
    load_model_checkpoint,
    predict_sr_from_raw_array,
    run_batch_inference_on_folder,
    safe_load_npy,
    save_final_outputs,
)
from src.models.simple_sr import SimpleSRNet
from src.config import SR_CNN_CHECKPOINT


def run_single_inference(input_path, output_dir, color_choice="cnn", save_dir=CHECKPOINT_DIR):
    raw = safe_load_npy(input_path)
    if raw.shape != (256, 256):
        raise ValueError(f"Expected raw TIR .npy shape (256, 256), got {raw.shape}")

    sr_model = load_model_checkpoint(
        SimpleSRNet(channels=64, num_blocks=6),
        os.path.join(save_dir, SR_CNN_CHECKPOINT),
    )
    color_model = load_best_color_model(color_choice, save_dir=save_dir)

    pred_tir = predict_sr_from_raw_array(sr_model, raw)
    pred_rgb = colorize_512_tir_by_tiles(color_model, pred_tir)

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    prefix = os.path.join(output_dir, f"{stem}_{color_choice}")
    return save_final_outputs(prefix, pred_tir, pred_rgb)


def main():
    parser = argparse.ArgumentParser(description="Final TIR SR + tile-wise colorization inference")
    parser.add_argument("--input", required=True, help="Raw TIR .npy file or folder of .npy files")
    parser.add_argument("--output", default="output/final_two_stage_outputs")
    parser.add_argument("--save-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--color-model", "--color_choice", dest="color_choice", choices=["cnn", "gan", "transformer"], default="cnn")
    parser.add_argument("--batch", action="store_true", help="Run folder batch inference and write inference_manifest.json")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        records = run_batch_inference_on_folder(
            args.input,
            args.output,
            color_choice=args.color_choice,
            max_files=args.max_files,
            save_dir=args.save_dir,
        )
        print(f"Batch inference complete: {len(records)} records")
    else:
        saved = run_single_inference(args.input, args.output, args.color_choice, args.save_dir)
        print(saved)


if __name__ == "__main__":
    main()
