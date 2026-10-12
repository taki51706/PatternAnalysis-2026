# Assessment requirement check

This is a tutor-facing check against the supplied **draft** COMP3710/COMP7710 brief. It records evidence, not an awarded grade.

| Brief requirement | Evidence in this project | Status |
| --- | --- | --- |
| Choose one approved project and difficulty | HipMRI 3D Improved U-Net; hard difficulty | Implemented |
| Baseline on same evaluation split | 2D U-Net; shared patient split and resampled grid | Implemented; validation-only real-data comparison measured |
| Modular `modules.py`, `dataset.py`, `train.py`, `predict.py` | All four files in this directory | Implemented |
| PyTorch model components; no NumPy in `modules.py` | `modules.py` imports only PyTorch and standard-library `math` | Checked |
| Patient-level train/validation/test separation | Explicit `patient_id` manifest, disjoint split validation, saved split | Synthetic test passed; native manifest has 211 volumes from 38 patients |
| Nifti alignment, intensity normalization, mask encoding | Affine and shape checks; canonical orientation; clipping and normalization; nearest-mask resampling; one-hot target in Dice loss | Native data audit passed for all 211 pairs; real-data CPU pilot and tuning runs completed |
| Training augmentation | Optional aligned left-right flip of image and masks in `dataset.py`, enabled by `train.py --augment` | Implemented; off in reported CPU runs, and no augmentation benefit has been measured |
| Prostate Dice target ≥0.70, per-class Dice/IoU | Patient-balanced and scan-weighted binary-class metrics; native-grid and resampled-grid test values after finalization | 10-epoch validation-only 3D patient-balanced Dice was 0.597 on the resampled grid; target not demonstrated |
| Boundary and edge-slice investigation | Apex/base Dice, false positives beyond gland extent, native-grid HD95 in millimeters, overlays | Preliminary validation metrics and pilot images exist; final-model case interpretation pending |
| Bladder/rectum leakage | Original anatomical labels retained; optional ID-specific false-positive counts | Course tutor confirmed native IDs 3/4 for bladder/rectum |
| Parameters, GPU memory, inference latency | `train.py` profile fields | GPU measurements pending |
| 3–5 real failure autopsies | `predict.py` selects low-Dice cases from different patients and creates five multi-panel images and index | Five exploratory pilot panels inspected; final-model review and clinically supported triggers pending |
| One-page feasibility review | Dedicated README section | Drafted |
| Compulsory midpoint tutor consultation | Brief requires a 3–5 minute review check-off | Not verified; this exemplar cannot stand in for an individual student's consultation |
| README report, figures, use instructions, versions, references, AI disclosure | README, Mermaid diagram, requirements, code map, measured validation table | Partially present; final test table, example output figure embedded in README, GPU profile, and conclusion pending |
| Progressive commit history | Local Git branch with staged milestone commits | In progress; cannot substitute for a student's own history |
| Pull request to upstream `topic-recognition` | Local branch based on that branch | Not submitted; tutor exemplar has no student fork |
| Turnitin PDF export of completed README | Requires completed real-data report | Pending |
| Respond to later code review | Requires teaching-team review after submission | Future event |

## Verified locally

- `pytest`: synthetic tests cover patient disjointness, singleton 4D Nifti support, tensor types, gradient flow, native-grid exact-match metrics, suffix-aware pairing, patient-balanced aggregation, and failure-case diversity. Rerun them after the latest code changes.
- One-epoch synthetic end-to-end run: both models trained, selected checkpoints, generated curves, evaluated the same held-out synthetic case, and wrote results/per-case files.
- `predict.py`: reloaded both checkpoints and generated a five-column, three-slice error figure. Its visual layout was inspected.
- The synthetic numerical scores are intentionally excluded from the README results table because they are not HipMRI results.
- The 211 native image/mask pairs passed an audit of shape, affine, and integer labels. A one-epoch CPU run trained and predicted on real data; its exploratory 3D model missed the prostate on the five displayed failure panels. A 10-epoch validation-only run reached patient-balanced resampled-grid Dice 0.606 (2D) and 0.597 (3D) on 34 visits from 6 patients. Neither is a held-out performance claim or evidence that the ≥0.70 target was reached.
- The first one-epoch pilot exposed this same test split. Any later result on it must be labeled an internal test comparison, not an untouched blind estimate.

## Required before describing this as a finished empirical submission

1. Choose architecture and training settings from validation data, complete a higher-resolution run on a suitable GPU, and evaluate the selected run once. Report that this particular test split was seen during the exploratory pilot. For a blind generalization estimate, obtain an independent patient cohort or prospective unseen holdout.
2. Inspect every generated metric and at least five final-model error panels; embed representative outputs in the README and write case-specific explanations, likely triggers, and an evidence-based conclusion. Measure target-GPU memory and latency.
3. Complete the midpoint tutor consultation check-off. Export the completed README to PDF, check rendering, and perform the course PR, Turnitin, and later code-review steps in the relevant student account and timeline.
