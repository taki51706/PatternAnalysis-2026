# Assessment requirement check

This is a tutor-facing check against the supplied **draft** COMP3710/COMP7710 brief. It records evidence, not an awarded grade.

| Brief requirement | Evidence in this project | Status |
| --- | --- | --- |
| Choose one approved project and difficulty | HipMRI 3D Improved U-Net; hard difficulty | Implemented |
| Baseline on same evaluation split | 2D U-Net; shared patient split and resampled grid | Implemented; real-data comparison pending |
| Modular `modules.py`, `dataset.py`, `train.py`, `predict.py` | All four files in this directory | Implemented |
| PyTorch model components; no NumPy in `modules.py` | `modules.py` imports only PyTorch and standard-library `math` | Checked |
| Patient-level train/validation/test separation | Explicit `patient_id` manifest, disjoint split validation, saved split | Synthetic test passed; supplied 2D slices have disjoint patient IDs across their provided splits |
| Nifti alignment, intensity normalization, mask encoding | Affine and shape checks; canonical orientation; clipping and normalization; nearest-mask resampling; one-hot target in Dice loss | Synthetic test passed; real data pending |
| Prostate Dice target ≥0.70, per-class Dice/IoU | `metrics.py`, `results.json` and `per_case.csv` | No real-data score yet |
| Boundary and edge-slice investigation | Apex/base Dice, 95th-percentile surface distance, false-positive/negative overlays | Code verified; clinical interpretation pending |
| Bladder/rectum leakage | Original anatomical labels retained; optional ID-specific false-positive counts | Awaiting verified label IDs and real data |
| Parameters, GPU memory, inference latency | `train.py` profile fields | GPU measurements pending |
| 3–5 real failure autopsies | `predict.py` creates five worst-case multi-panel images and index | Real cases and written diagnoses pending |
| One-page feasibility review | Dedicated README section | Drafted |
| README report, figures, use instructions, versions, references, AI disclosure | README, Mermaid diagram, requirements, code map | Present; empirical report pending |
| Progressive commit history | Local Git branch with staged milestone commits | In progress; cannot substitute for a student's own history |
| Pull request to upstream `topic-recognition` | Local branch based on that branch | Not submitted; tutor exemplar has no student fork |
| Turnitin PDF export of completed README | Requires completed real-data report | Pending |
| Respond to later code review | Requires teaching-team review after submission | Future event |

## Verified locally

- `pytest`: 2 synthetic tests pass, including patient disjointness, singleton 4D Nifti support, tensor types, gradient flow, and exact-match metrics.
- One-epoch synthetic end-to-end run: both models trained, selected checkpoints, generated curves, evaluated the same held-out synthetic case, and wrote results/per-case files.
- `predict.py`: reloaded both checkpoints and generated a five-column, three-slice error figure. Its visual layout was inspected.
- The synthetic numerical scores are intentionally excluded from the README results table because they are not HipMRI results.

## Required before describing this as a finished empirical submission

1. Obtain the native 3D HipMRI folders for this chosen project, or explicitly adapt it to a 2D or derived slice-stack study. Verify the prostate, bladder, and rectum label IDs before training.
2. Run the locked experiment on Rangpur, inspect every generated metric and at least five real error panels, and write case-specific explanations and a conclusion supported by those findings.
3. Export the completed README to PDF, check rendering, and perform the course PR, Turnitin, and later code-review steps in the relevant student account and timeline.
