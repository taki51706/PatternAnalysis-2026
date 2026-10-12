# COMP3710 medical image analysis exemplar

The project is in [`recognition/hipmri_resunet3d_tutor_example`](recognition/hipmri_resunet3d_tutor_example/README.md). It is built on the course repository's `topic-recognition` branch in the local `tutor-example/hipmri-3d` branch.

- Read the [project report and run instructions](recognition/hipmri_resunet3d_tutor_example/README.md).
- Read the [assessment requirement check](recognition/hipmri_resunet3d_tutor_example/ASSESSMENT_CHECK.md).
- Read the [exploratory native-data pilot notes](recognition/hipmri_resunet3d_tutor_example/PILOT_NOTES.md) separately from the selected-run results.
- The supplied CSIRO release contains 211 paired native 3D volumes from 38 patients; the pair audit passed. Four CPU candidates were compared using validation data. The selected 3D model reached patient-balanced Dice 0.694 after restoration to the native grid versus 0.549 for the 2D baseline, on 23 visits from five test patients. The coarser network-grid figures were 0.757 and 0.610. The brief's downsampled 0.70 Dice target is met; the stricter native-grid mean remains below 0.70.
- The early one-epoch pilot exposed this same test split before the validation-only gate was added. The selected-run scores are therefore an **internal exploratory comparison**, not a blind estimate. The README reports native boundary and organ-leakage metrics, patient-bootstrap uncertainty, and five inspected case panels, including a case where 2D wins.
- The tutor exemplar has a completed local report, checked source, and archived run evidence. A higher-resolution GPU benchmark, target-GPU memory profile, independent patient cohort, and specialist/dosimetric assessment are separate scientific gates.
- The Git history shows development milestones for this tutor exemplar. Students need their own implementation and history, AI disclosure, consultation check-off, pull request, and Turnitin submission.
