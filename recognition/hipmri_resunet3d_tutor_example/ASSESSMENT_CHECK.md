# Assessment requirement check

This tutor-facing audit compares the exemplar with the supplied **draft** COMP3710/COMP7710 brief. It records evidence, not an awarded grade or a student's submission.

| Brief requirement | Evidence | Status |
| --- | --- | --- |
| Approved project and difficulty | HipMRI medical image analysis; hard 3D Improved U-Net | Implemented |
| Same-data baseline | Slice-wise 2D U-Net and 3D model use the same saved patient split, image grid, and mask encoding | Measured on validation and internal test |
| Modular `modules.py`, `dataset.py`, `train.py`, `predict.py` | All files present; `modules.py` uses PyTorch and standard-library `math` only | Checked |
| Patient-level train/validation/test split | 27/6/5 disjoint patients; 154/34/23 visits; saved `split.json` | Checked |
| Nifti alignment, normalization, labels | All 211 native image/mask pairs passed shape, affine, and integer-label audit; course tutor confirmed IDs 5/3/4 | Checked |
| Training augmentation | Aligned left-right flip option in `dataset.py`/`train.py` | Implemented; off in measured runs |
| Dice target ≥0.70 and per-class Dice/IoU | Selected 3D internal test: resampled patient mean 0.757 and scan mean 0.737; native patient mean 0.694; background/prostate Dice and IoU recorded | Brief's downsampled target met; stricter native 0.70 check not met |
| Boundary and edge-slice analysis | Native patient mean apex 0.348, base 0.636, HD95 8.17 mm; five final-model panels | Measured, with small-cohort uncertainty |
| Bladder/rectum leakage | Native patient mean false-positive fractions 0.97% and 0.261% for 3D, lower than the 2D baseline | Measured |
| Parameters, memory, inference latency | 3D: 93,358 parameters and 74 ms model-only CPU inference per volume | Target-GPU memory and latency unmeasured |
| Three to five failure autopsies | Five panels from different patients, with observed false positives/negatives and written case reviews | Completed; no specialist adjudication |
| One-page feasibility review | Dedicated README section with need, goals, risks, budget, and fallback | Completed |
| README report, figures, versions, references, AI disclosure | Completed README, patient Dice plot, representative error panel, code map, exact CPU configuration, measured tables | Completed as tutor exemplar |
| Progressive commit history | Local `tutor-example/hipmri-3d` branch with milestone commits | Present; not a student's own history |
| Midpoint tutor consultation | Brief calls for a 3–5 minute review/check-off | Requires student's own attendance |
| Pull request to upstream `topic-recognition` | Exemplar is on a local branch, with no student fork | Requires student account and course workflow |
| Turnitin PDF | A rendered PDF of the completed README is supplied separately | Tutor demonstration only; student submission remains theirs |
| Respond to later code review | Depends on teaching-team review after submission | Future event |

## Local verification

- The automated suite has 18 synthetic tests covering split disjointness, singleton 4D Nifti support, alignment, tensor shapes, gradient flow, native-grid metrics, suffix-aware pairing, patient-balanced aggregation, one-time test gate, and case-selection diversity. Synthetic scores are never presented as HipMRI performance.
- The native audit passed for 211 pairs from 38 patients. A train/validation-only crop check confirmed all 188 prostates lie within a central 128×128 crop; the crop candidates were not selected because their 3D validation Dice was lower.
- Four candidates were compared using validation data only; the selected full-field weight-20 run reached 3D validation Dice 0.684. Its one-time internal test on 23 visits yielded native patient-balanced Dice 0.694 versus 0.549 for 2D. The paired five-patient bootstrap difference interval was +0.020 to +0.283, which is descriptive for this small cohort.
- All five final-model case panels were inspected. They include a patient for whom 2D outperformed 3D and apex omissions despite good whole-volume overlap.
- The **earlier one-epoch pilot exposed this same test split** before the validation-only gate existed. All subsequent scores on it are internal exploratory comparisons. A genuinely blind estimate requires a separate independent cohort.

## Remaining course or scientific gates

1. For a full-resolution claim, run a separately selected higher-resolution experiment on a suitable GPU and measure target-GPU memory and latency. The reported CPU result belongs to a coarse grid and small model.
2. For generalization and clinical usefulness, evaluate an unseen patient cohort, obtain contour specialist adjudication, and assess native-space dosimetric impact. No automatic clinical use is supported by this exemplar.
3. Each student must complete the tutor check-off, independent implementation/history, course pull request, Turnitin submission, and later review response in the required account and timeline.
