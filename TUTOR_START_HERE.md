# COMP3710 medical image analysis exemplar

The project is in [`recognition/hipmri_resunet3d_tutor_example`](recognition/hipmri_resunet3d_tutor_example/README.md). It is built on the course repository's `topic-recognition` branch in the local `tutor-example/hipmri-3d` branch.

- Read the [project report and run instructions](recognition/hipmri_resunet3d_tutor_example/README.md).
- Read the [assessment requirement check](recognition/hipmri_resunet3d_tutor_example/ASSESSMENT_CHECK.md).
- Read the [exploratory native-data pilot notes](recognition/hipmri_resunet3d_tutor_example/PILOT_NOTES.md) separately from final results.
- The supplied CSIRO release contains 211 paired native 3D volumes from 38 patients. The manifest and image/mask audit passed. A one-epoch CPU pilot ran on real data and a 10-epoch, validation-only CPU tuning run measured preliminary scores. The 3D model reached patient-balanced prostate Dice 0.597 on the small resampled validation grid; this is below the brief's 0.70 test target and is not a test result.
- The early one-epoch pilot exposed this patient split's test results before the validation-only gate was added. Later evaluation on the same split is an internal comparison, not an untouched blind benchmark. The project README gives the validation-only training and one-time `evaluate_test.py` workflow.
- A higher-resolution GPU run, target-GPU memory profile, final-model case review, and evidence-based conclusion are still needed before calling this a finished empirical report.
- The Git history shows development milestones for this tutor exemplar. Students need their own implementation and their own commit history, AI disclosure, pull request, and Turnitin submission.
