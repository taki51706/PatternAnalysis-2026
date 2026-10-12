"""Train 2D baseline and residual 3D model on identical patient splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    SliceDataset,
    VolumeDataset,
    load_split,
    read_manifest,
    save_split,
    split_by_patient,
)
from metrics import case_metrics
from modules import ImprovedUNet3D, UNet2D


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross_entropy = F.cross_entropy(logits, target)
    probabilities = logits.softmax(dim=1)[:, 1]
    foreground = F.one_hot(target, num_classes=2).movedim(-1, 1)[:, 1].float()
    spatial_dims = tuple(range(1, foreground.ndim))
    intersection = (probabilities * foreground).sum(dim=spatial_dims)
    predicted_area = probabilities.sum(dim=spatial_dims)
    target_area = foreground.sum(dim=spatial_dims)
    soft_dice = (2 * intersection + 1) / (predicted_area + target_area + 1)
    return cross_entropy + 1 - soft_dice.mean()


def model_loss(output: torch.Tensor | dict, target: torch.Tensor) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return segmentation_loss(output, target)
    loss = segmentation_loss(output["logits"], target)
    for weight, auxiliary_logits in zip((0.25, 0.125), output["aux"]):
        scaled_target = F.interpolate(
            target[:, None].float(), size=auxiliary_logits.shape[2:], mode="nearest"
        )[:, 0].long()
        loss += weight * segmentation_loss(auxiliary_logits, scaled_target)
    return loss


@torch.no_grad()
def predict_volume(model: torch.nn.Module, image: torch.Tensor, kind: str,
                   device: torch.device, slice_batch: int = 16) -> np.ndarray:
    """Run either model on one [1,D,H,W] image and return a [D,H,W] mask."""
    model.eval()
    if kind == "2d":
        image_slices = image[0].unsqueeze(1)  # [D,1,H,W]
        predicted_slices = []
        for batch in image_slices.split(slice_batch):
            predicted_slices.append(model(batch.to(device)).argmax(dim=1).cpu())
        return torch.cat(predicted_slices).numpy().astype(bool)

    logits = model(image[None].to(device))["logits"]
    return logits.argmax(dim=1)[0].cpu().numpy().astype(bool)


def _mean_case_metrics(rows: list[dict]) -> dict:
    """Average metrics over cases while preserving the number evaluated."""
    summary = {}
    metric_names = set(rows[0]) - {"case_id", "patient_id"}
    for name in sorted(metric_names):
        values = [row[name] for row in rows if row[name] is not None]
        summary[name] = float(np.mean(values)) if values else None
    summary["cases"] = len(rows)
    summary["patients"] = len({row["patient_id"] for row in rows})
    return summary


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: VolumeDataset, kind: str,
             device: torch.device, bladder_label: int | None = None,
             rectum_label: int | None = None) -> tuple[dict, list[dict]]:
    rows = []
    for sample in dataset:
        sync(device)
        started = time.perf_counter()
        pred = predict_volume(model, sample["image"], kind, device)
        sync(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics = case_metrics(
            pred,
            sample["mask"].numpy(),
            sample["anatomy"].numpy(),
            bladder_label,
            rectum_label,
        )
        rows.append({
            "case_id": sample["case_id"],
            "patient_id": sample["patient_id"],
            "latency_ms": elapsed_ms,
            **metrics,
        })
    return _mean_case_metrics(rows), rows


def _training_loader(
    kind: str, volumes: VolumeDataset, args: argparse.Namespace
) -> DataLoader:
    """Use balanced slice sampling for 2D and whole volumes for 3D."""
    if kind == "2d":
        slices = SliceDataset(volumes)
        sampler = WeightedRandomSampler(
            slices.weights, num_samples=len(slices), replacement=True
        )
        return DataLoader(slices, batch_size=args.batch_2d, sampler=sampler)
    return DataLoader(volumes, batch_size=args.batch_3d, shuffle=True)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model(image), mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _plot_history(history: list[dict], path: Path) -> None:
    epochs = [record["epoch"] for record in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [record["train_loss"] for record in history])
    axes[0].set(xlabel="Epoch", ylabel="CE + Dice loss", title="Training loss")

    for label, key in (
        ("whole volume", "val_prostate_dice"),
        ("apex", "val_apex_dice"),
        ("base", "val_base_dice"),
    ):
        axes[1].plot(epochs, [record[key] for record in history], label=label)
    axes[1].set(xlabel="Epoch", ylabel="Dice", ylim=(0, 1), title="Validation")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one(
    kind: str,
    train_volumes: VolumeDataset,
    val_volumes: VolumeDataset,
    device: torch.device,
    out: Path,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict]:
    """Fit one model and restore its best validation checkpoint."""
    width = args.width_2d if kind == "2d" else args.width_3d
    model = UNet2D(width=width) if kind == "2d" else ImprovedUNet3D(width=width)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = _training_loader(kind, train_volumes, args)

    history = []
    best_score = -1.0
    stale_epochs = 0
    checkpoint = out / f"{kind}_best.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, loader, optimizer, device)
        validation, _ = evaluate(
            model, val_volumes, kind, device, args.bladder_label, args.rectum_label
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_prostate_dice": validation["prostate_dice"],
            "val_apex_dice": validation["apex_dice"],
            "val_base_dice": validation["base_dice"],
        }
        history.append(record)
        print(json.dumps({"model": kind, **record}), flush=True)

        if validation["prostate_dice"] > best_score + 1e-5:
            best_score = validation["prostate_dice"]
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "kind": kind,
                "width": width,
                "size": args.size,
                "prostate_label": args.prostate_label,
                "best_epoch": epoch,
            }, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    best_checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])
    (out / f"{kind}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    _plot_history(history, out / f"{kind}_training_curves.png")
    profile = {
        "best_epoch": best_checkpoint["best_epoch"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_device_memory_mb": memory_mb(device),
    }
    return model, profile


def parse_arguments() -> argparse.Namespace:
    """Define the experiment configuration in one place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--prostate-label", type=int, required=True,
        help="Inspect original mask IDs first; never assume the prostate label",
    )
    parser.add_argument("--bladder-label", type=int, default=None)
    parser.add_argument("--rectum-label", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/hipmri"))
    parser.add_argument(
        "--size", nargs=3, type=int, default=[64, 128, 128],
        metavar=("D", "H", "W"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-2d", type=int, default=16)
    parser.add_argument("--batch-3d", type=int, default=1)
    parser.add_argument("--width-2d", type=int, default=16)
    parser.add_argument("--width-3d", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    chosen_labels = [
        label for label in (
            args.prostate_label, args.bladder_label, args.rectum_label
        ) if label is not None
    ]
    if len(set(chosen_labels)) != len(chosen_labels):
        parser.error("Prostate, bladder, and rectum IDs must be distinct")
    return args


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    cases = read_manifest(args.manifest)
    split_path = args.out / "split.json"
    if split_path.exists():
        split = load_split(cases, split_path)
    else:
        split = split_by_patient(cases, args.seed)
        save_split(split, split_path)

    size = tuple(args.size)
    train_data = VolumeDataset(split["train"], size, args.prostate_label)
    val_data = VolumeDataset(split["val"], size, args.prostate_label)
    test_data = VolumeDataset(split["test"], size, args.prostate_label)
    device = choose_device(args.device)
    split_counts = {name: len(group) for name, group in split.items()}
    print(f"device={device}; cases={split_counts}", flush=True)

    results = {}
    trained_models = {}
    for kind in ("2d", "3d"):
        model, profile = train_one(kind, train_data, val_data, device, args.out, args)
        validation, _ = evaluate(
            model, val_data, kind, device, args.bladder_label, args.rectum_label
        )
        results[kind] = {"profile": profile, "validation": validation}
        trained_models[kind] = model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # The held-out test split is opened only after both models are selected.
    all_rows = []
    for kind in ("2d", "3d"):
        model = trained_models[kind].to(device)
        test, rows = evaluate(
            model, test_data, kind, device, args.bladder_label, args.rectum_label
        )
        results[kind]["test"] = test
        all_rows.extend({"model": kind, **row} for row in rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.out / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    results["provenance"] = {
        "manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "size_dhw": args.size,
        "prostate_label": args.prostate_label,
        "bladder_label": args.bladder_label,
        "rectum_label": args.rectum_label,
        "torch": torch.__version__,
        "device": str(device),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
