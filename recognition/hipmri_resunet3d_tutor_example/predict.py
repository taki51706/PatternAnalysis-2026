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
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = (UNet2D(width=checkpoint["width"]) if checkpoint["kind"] == "2d"
             else ImprovedUNet3D(width=checkpoint["width"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def draw_case(sample: dict, pred_2d: np.ndarray, pred_3d: np.ndarray, path: Path) -> None:
    image = sample["image"][0].numpy()
    truth = sample["mask"].numpy().astype(bool)
    active = np.flatnonzero(truth.any(axis=(1, 2)))
    slices = [int(active[0]), int(active[len(active) // 2]), int(active[-1])]
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    titles = ["MRI input", "Expert contour", "2D U-Net", "3D Improved U-Net", "3D errors: FP red / FN blue"]
    for row, z in enumerate(slices):
        gray = image[z]
        for col, ax in enumerate(axes[row]):
            ax.imshow(gray, cmap="gray", vmin=np.percentile(gray, 1), vmax=np.percentile(gray, 99))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
        for col, mask in ((1, truth[z]), (2, pred_2d[z]), (3, pred_3d[z])):
            if mask.any():
                axes[row, col].contour(mask, levels=[0.5], colors=["#00e5ff"], linewidths=1.5)
        error = np.zeros((*gray.shape, 4), dtype=float)
        error[np.logical_and(pred_3d[z], ~truth[z])] = (1, 0, 0, 0.65)
        error[np.logical_and(~pred_3d[z], truth[z])] = (0, 0.4, 1, 0.65)
        axes[row, 4].imshow(error)
        axes[row, 0].set_ylabel(("apex", "mid", "base")[row] + f"  z={z}")
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
    if (base_config["size"] != new_config["size"] or
            base_config["prostate_label"] != new_config["prostate_label"]):
        raise ValueError("Models were trained with different preprocessing")
    split = load_split(read_manifest(args.manifest), args.run_dir / "split.json")
    test = VolumeDataset(split["test"], tuple(new_config["size"]), new_config["prostate_label"])
    evaluated = []
    for sample in test:
        pred_2d = predict_volume(baseline, sample["image"], "2d", device)
        pred_3d = predict_volume(improved, sample["image"], "3d", device)
        truth = sample["mask"].numpy()
        metrics_2d, metrics_3d = case_metrics(pred_2d, truth), case_metrics(pred_3d, truth)
        evaluated.append((metrics_3d["prostate_dice"], sample, pred_2d, pred_3d, metrics_2d, metrics_3d))
    evaluated.sort(key=lambda item: item[0])
    rows = []
    for rank, (_, sample, pred_2d, pred_3d, baseline_metrics, improved_metrics) in enumerate(evaluated[:args.count], 1):
        name = f"{rank:02d}_{sample['case_id']}.png"
        draw_case(sample, pred_2d, pred_3d, out / name)
        rows.append({"rank": rank, "case_id": sample["case_id"], "patient_id": sample["patient_id"],
                     "figure": name, "baseline_dice": baseline_metrics["prostate_dice"],
                     "improved_dice": improved_metrics["prostate_dice"],
                     "improved_apex_dice": improved_metrics["apex_dice"],
                     "improved_base_dice": improved_metrics["base_dice"],
                     "false_positive_voxels": improved_metrics["false_positive_voxels"],
                     "false_negative_voxels": improved_metrics["false_negative_voxels"]})
        print(f"{name}: 2D Dice={baseline_metrics['prostate_dice']:.3f}; 3D Dice={improved_metrics['prostate_dice']:.3f}")
    with (out / "autopsy_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
