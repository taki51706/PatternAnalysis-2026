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

from dataset import SliceDataset, VolumeDataset, load_split, read_manifest, save_split, split_by_patient
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
    ce = F.cross_entropy(logits, target)
    probabilities = logits.softmax(dim=1)[:, 1]
    foreground = (target == 1).float()
    dims = tuple(range(1, foreground.ndim))
    intersection = (probabilities * foreground).sum(dim=dims)
    dice = (2 * intersection + 1) / (probabilities.sum(dim=dims) + foreground.sum(dim=dims) + 1)
    return ce + 1 - dice.mean()


def model_loss(output: torch.Tensor | dict, target: torch.Tensor) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return segmentation_loss(output, target)
    loss = segmentation_loss(output["logits"], target)
    for weight, aux in zip((0.25, 0.125), output["aux"]):
        scaled = F.interpolate(target[:, None].float(), size=aux.shape[2:], mode="nearest")[:, 0].long()
        loss = loss + weight * segmentation_loss(aux, scaled)
    return loss


@torch.no_grad()
def predict_volume(model: torch.nn.Module, image: torch.Tensor, kind: str,
                   device: torch.device, slice_batch: int = 16) -> np.ndarray:
    model.eval()
    if kind == "2d":
        slices = image[0].unsqueeze(1)  # [D,1,H,W]
        outputs = []
        for batch in slices.split(slice_batch):
            outputs.append(model(batch.to(device)).argmax(dim=1).cpu())
        return torch.cat(outputs).numpy().astype(bool)
    output = model(image[None].to(device))["logits"]
    return output.argmax(dim=1)[0].cpu().numpy().astype(bool)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: VolumeDataset, kind: str,
             device: torch.device) -> tuple[dict, list[dict]]:
    rows = []
    for sample in dataset:
        sync(device)
        started = time.perf_counter()
        pred = predict_volume(model, sample["image"], kind, device)
        sync(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics = case_metrics(pred, sample["mask"].numpy())
        rows.append({"case_id": sample["case_id"], "patient_id": sample["patient_id"],
                     "latency_ms": elapsed_ms, **metrics})
    numeric = [key for key in rows[0] if key not in ("case_id", "patient_id")]
    summary = {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
               if any(row[key] is not None for row in rows) else None for key in numeric}
    summary["cases"] = len(rows)
    summary["patients"] = len({row["patient_id"] for row in rows})
    return summary, rows


def train_one(kind: str, train_volumes: VolumeDataset, val_volumes: VolumeDataset,
              device: torch.device, out: Path, args: argparse.Namespace) -> tuple[torch.nn.Module, dict]:
    width = args.width_2d if kind == "2d" else args.width_3d
    model = (UNet2D(width=width) if kind == "2d" else ImprovedUNet3D(width=width)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if kind == "2d":
        slices = SliceDataset(train_volumes)
        sampler = WeightedRandomSampler(slices.weights, num_samples=len(slices), replacement=True)
        loader = DataLoader(slices, batch_size=args.batch_2d, sampler=sampler, num_workers=0)
    else:
        loader = DataLoader(train_volumes, batch_size=args.batch_3d, shuffle=True, num_workers=0)
    history, best_score, stale = [], -1.0, 0
    checkpoint = out / f"{kind}_best.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model_loss(model(image), mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val, _ = evaluate(model, val_volumes, kind, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "val_prostate_dice": val["prostate_dice"],
                  "val_apex_dice": val["apex_dice"], "val_base_dice": val["base_dice"]}
        history.append(record)
        print(json.dumps({"model": kind, **record}), flush=True)
        if val["prostate_dice"] > best_score + 1e-5:
            best_score, stale = val["prostate_dice"], 0
            torch.save({"state_dict": model.state_dict(), "kind": kind, "width": width,
                        "size": args.size, "prostate_label": args.prostate_label,
                        "best_epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    (out / f"{kind}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history])
    axes[0].set(xlabel="Epoch", ylabel="CE + Dice loss", title="Training loss")
    axes[1].plot([h["epoch"] for h in history], [h["val_prostate_dice"] for h in history], label="whole volume")
    axes[1].plot([h["epoch"] for h in history], [h["val_apex_dice"] for h in history], label="apex")
    axes[1].plot([h["epoch"] for h in history], [h["val_base_dice"] for h in history], label="base")
    axes[1].set(xlabel="Epoch", ylabel="Dice", ylim=(0, 1), title="Validation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / f"{kind}_training_curves.png", dpi=160)
    plt.close(fig)
    return model, {"best_epoch": torch.load(checkpoint, weights_only=True)["best_epoch"],
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "peak_device_memory_mb": memory_mb(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prostate-label", type=int, required=True,
                        help="Inspect original mask IDs first; never assume the prostate label")
    parser.add_argument("--out", type=Path, default=Path("runs/hipmri"))
    parser.add_argument("--size", nargs=3, type=int, default=[64, 128, 128],
                        metavar=("D", "H", "W"))
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
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cases = read_manifest(args.manifest)
    split_path = args.out / "split.json"
    split = load_split(cases, split_path) if split_path.exists() else split_by_patient(cases, args.seed)
    if not split_path.exists():
        save_split(split, split_path)
    size = tuple(args.size)
    train_data = VolumeDataset(split["train"], size, args.prostate_label)
    val_data = VolumeDataset(split["val"], size, args.prostate_label)
    test_data = VolumeDataset(split["test"], size, args.prostate_label)
    device = choose_device(args.device)
    print(f"device={device}; cases=" + str({name: len(v) for name, v in split.items()}), flush=True)
    results, all_rows = {}, []
    for kind in ("2d", "3d"):
        model, profile = train_one(kind, train_data, val_data, device, args.out, args)
        validation, _ = evaluate(model, val_data, kind, device)
        test, rows = evaluate(model, test_data, kind, device)
        results[kind] = {"profile": profile, "validation": validation, "test": test}
        all_rows.extend({"model": kind, **row} for row in rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (args.out / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    results["provenance"] = {"manifest": str(Path(args.manifest).resolve()),
                             "seed": args.seed, "size_dhw": args.size,
                             "prostate_label": args.prostate_label,
                             "torch": torch.__version__, "device": str(device)}
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
