"""Load saved weights and visualize five held-out failure cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import VolumeDataset, load_split, read_manifest
from metrics import case_metrics
from modules import ImprovedUNet3D, UNet2D
from train import choose_device, predict_volume


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Recreate the architecture recorded in a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["kind"] == "2d":
        model = UNet2D(width=checkpoint["width"])
    else:
        model = ImprovedUNet3D(width=checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def _error_overlay(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Show false positives in red and false negatives in blue."""
    overlay = np.zeros((*truth.shape, 4), dtype=float)
    overlay[np.logical_and(prediction, ~truth)] = (1, 0, 0, 0.65)
    overlay[np.logical_and(~prediction, truth)] = (0, 0.4, 1, 0.65)
    return overlay


def draw_case(sample: dict, pred_2d: np.ndarray, pred_3d: np.ndarray, path: Path) -> None:
    """Save apex, middle, and base slices for one held-out case."""
    image = sample["image"][0].numpy()
    truth = sample["mask"].numpy().astype(bool)
    prostate_slices = np.flatnonzero(truth.any(axis=(1, 2)))
    slices = [
        int(prostate_slices[0]),
        int(prostate_slices[len(prostate_slices) // 2]),
        int(prostate_slices[-1]),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    titles = [
        "MRI input", "Expert contour", "2D U-Net", "3D Improved U-Net",
        "3D errors: FP red / FN blue",
    ]
    for row, slice_index in enumerate(slices):
        gray = image[slice_index]
        vmin, vmax = np.percentile(gray, (1, 99))
        for col, ax in enumerate(axes[row]):
            ax.imshow(gray, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)

        for col, mask in (
            (1, truth[slice_index]),
            (2, pred_2d[slice_index]),
            (3, pred_3d[slice_index]),
        ):
            if mask.any():
                axes[row, col].contour(mask, levels=[0.5], colors=["#00e5ff"], linewidths=1.5)

        axes[row, 4].imshow(_error_overlay(pred_3d[slice_index], truth[slice_index]))
        axes[row, 0].set_ylabel(("apex", "mid", "base")[row] + f"  z={slice_index}")
    fig.suptitle(f"Held-out case {sample['case_id']} | patient {sample['patient_id']}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    out = args.out or args.run_dir / "failure_cases"
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    baseline, base_config = load_model(args.run_dir / "2d_best.pt", device)
    improved, new_config = load_model(args.run_dir / "3d_best.pt", device)
    if (
        base_config["size"] != new_config["size"]
        or base_config["prostate_label"] != new_config["prostate_label"]
    ):
        raise ValueError("Models were trained with different preprocessing")
    split = load_split(read_manifest(args.manifest), args.run_dir / "split.json")
    test = VolumeDataset(
        split["test"], tuple(new_config["size"]), new_config["prostate_label"]
    )
    evaluated = []
    for sample in test:
        pred_2d = predict_volume(baseline, sample["image"], "2d", device)
        pred_3d = predict_volume(improved, sample["image"], "3d", device)
        truth = sample["mask"].numpy()
        evaluated.append({
            "sample": sample,
            "pred_2d": pred_2d,
            "pred_3d": pred_3d,
            "baseline_metrics": case_metrics(pred_2d, truth),
            "improved_metrics": case_metrics(pred_3d, truth),
        })
    evaluated.sort(key=lambda item: item["improved_metrics"]["prostate_dice"])

    rows = []
    for rank, case in enumerate(evaluated[:args.count], 1):
        sample = case["sample"]
        baseline_metrics = case["baseline_metrics"]
        improved_metrics = case["improved_metrics"]
        name = f"{rank:02d}_{sample['case_id']}.png"
        draw_case(sample, case["pred_2d"], case["pred_3d"], out / name)
        rows.append({
            "rank": rank,
            "case_id": sample["case_id"],
            "patient_id": sample["patient_id"],
            "figure": name,
            "baseline_dice": baseline_metrics["prostate_dice"],
            "improved_dice": improved_metrics["prostate_dice"],
            "improved_apex_dice": improved_metrics["apex_dice"],
            "improved_base_dice": improved_metrics["base_dice"],
            "false_positive_voxels": improved_metrics["false_positive_voxels"],
            "false_negative_voxels": improved_metrics["false_negative_voxels"],
        })
        print(
            f"{name}: 2D Dice={baseline_metrics['prostate_dice']:.3f}; "
            f"3D Dice={improved_metrics['prostate_dice']:.3f}"
        )
    with (out / "autopsy_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
