"""Small synthetic checks for alignment, leakage, model shapes, and gradients."""

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import VolumeDataset, read_manifest, split_by_patient  # noqa: E402
from metrics import case_metrics  # noqa: E402
from modules import ImprovedUNet3D, UNet2D  # noqa: E402
from train import model_loss, predict_volume  # noqa: E402


def synthetic_manifest(tmp_path: Path) -> Path:
    rows = []
    for i in range(3):
        image = np.zeros((16, 16, 16), dtype=np.float32)
        image[2:14, 2:14, 2:14] = 1.0 + i
        image[6:10, 6:10, 5:11] = 2.0 + i
        mask = np.zeros_like(image)
        mask[6:10, 6:10, 5:11] = 5
        image_path, mask_path = tmp_path / f"p{i}_image.nii.gz", tmp_path / f"p{i}_mask.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(mask[..., None] if i == 0 else mask, np.eye(4)), mask_path)
        rows.append({"case_id": f"p{i}", "patient_id": f"patient_{i}",
                     "image": image_path.name, "mask": mask_path.name})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_patient_split_and_alignment(tmp_path: Path) -> None:
    cases = read_manifest(synthetic_manifest(tmp_path))
    split = split_by_patient(cases)
    groups = [set(c.patient_id for c in split[name]) for name in ("train", "val", "test")]
    assert all(groups)
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    sample = VolumeDataset(cases[:1], (16, 16, 16), 5)[0]
    assert sample["image"].shape == (1, 16, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].sum() == 4 * 4 * 6
    assert set(torch.unique(sample["mask"]).tolist()) == {0, 1}
    with pytest.raises(ValueError, match="absent"):
        VolumeDataset(cases[:1], (16, 16, 16), 4)[0]


def test_models_loss_prediction_and_metrics() -> None:
    torch.set_num_threads(2)
    volume = torch.randn(1, 1, 16, 16, 16)
    mask = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    mask[:, 5:11, 5:11, 5:11] = 1
    advanced = ImprovedUNet3D(width=4)
    output = advanced(volume)
    assert output["logits"].shape == (1, 2, 16, 16, 16)
    loss = model_loss(output, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert advanced.head.weight.grad is not None
    baseline = UNet2D(width=4)
    assert baseline(volume[:, :, 0]).shape == (1, 2, 16, 16)
    pred = predict_volume(advanced, volume[0], "3d", torch.device("cpu"))
    assert pred.shape == (16, 16, 16)
    perfect = case_metrics(mask[0].numpy(), mask[0].numpy())
    assert perfect["prostate_dice"] == 1.0
    assert perfect["hd95_resampled_voxels"] == 0.0
    anatomy = np.zeros((16, 16, 16), dtype=int)
    anatomy[0, 0, :4] = 3
    pred_with_leak = mask[0].numpy().copy()
    pred_with_leak[0, 0, 0] = 1
    leakage = case_metrics(pred_with_leak, mask[0].numpy(), anatomy, bladder_label=3)
    assert leakage["false_positive_bladder_voxels"] == 1
    assert leakage["false_positive_bladder_fraction"] == 0.25
