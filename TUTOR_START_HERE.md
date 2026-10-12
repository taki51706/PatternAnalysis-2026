# COMP3710 medical image analysis exemplar

The project is in [`recognition/hipmri_resunet3d_tutor_example`](recognition/hipmri_resunet3d_tutor_example/README.md). It is built on the course repository's `topic-recognition` branch in the local `tutor-example/hipmri-3d` branch.

- Read the [project report and run instructions](recognition/hipmri_resunet3d_tutor_example/README.md).
- Read the [assessment requirement check](recognition/hipmri_resunet3d_tutor_example/ASSESSMENT_CHECK.md).
- Read the [grade-readiness audit](recognition/hipmri_resunet3d_tutor_example/GRADE_READINESS.md) for the 40-mark rubric and student-specific pass hurdles.
- Read the [exploratory native-data pilot notes](recognition/hipmri_resunet3d_tutor_example/PILOT_NOTES.md) separately from the selected-run results.
- The supplied CSIRO release contains 211 paired native 3D volumes from 38 patients; the pair audit passed. Four CPU candidates used the same patient split. On 23 visits from five test patients, the selected 3D model reached **0.694** patient-balanced native Dice, compared with **0.549** for the matched full-field 2D model and **0.698** for the independently validation-selected cropped 2D model. The 3D model scored 0.757 on its coarse network grid, meeting the brief's downsampled 0.70 target, but it has no native Dice advantage over the stronger 2D comparator.
- Against that stronger 2D comparator, 3D native apex/base Dice was slightly lower (0.348/0.636 versus 0.388/0.645). Its HD95 and outside-extent false positives were lower, while model-only CPU inference took about 22 ms longer. The five case panels compare the common full-field pair, not the separately tuned cropped 2D model.
- The early one-epoch pilot exposed this same test split before the validation-only gate was added. Both selected checkpoints were subsequently chosen using validation data and each received one recorded test evaluation, but all test scores remain **internal exploratory comparisons**, not blind estimates. The README reports the mixed boundary and organ-leakage results and a paired five-patient Dice difference interval that crosses zero.
- The tutor exemplar has a completed local report, checked source, and saved run evidence. A higher-resolution GPU benchmark, target-GPU memory profile, independent patient cohort, and specialist/dosimetric assessment are separate scientific gates.
- The Git history shows development milestones for this tutor exemplar. Students need their own implementation and history, AI disclosure, consultation check-off, pull request, and Turnitin submission.
