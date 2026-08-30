# Dahua Cup competition application

This directory contains the project-owned code for the campus behavior
recognition competition:

- `feature_extraction/`: MediaPipe pose extraction, filtering, graph building,
  quality features, and pose rendering.
- `person_localization/`: NanoDet person-only inference and deterministic
  zero/one/two-person group cropping before MediaPipe.
- `pipeline/`: dataset extraction, student/teacher workers, pseudo-label
  generation, training launchers, and closed-loop orchestration.
- `semantic_teacher/`: Qwen3-VL teacher, prompts, schemas, filtering, review,
  distillation, and incremental-learning components.
- `backend/` and `visualization/`: remote inference, human review, and Web UI.
- `configs/`, `scripts/`, and `tests/`: configuration, operational entry
  points, and regression coverage.

Run modules from the repository root with the `dahua_cup.*` prefix. The GCN
repositories are siblings under `../gcn_models/`; datasets, checkpoints, and
experiment outputs remain under the configured external data root.

The complete Campus6 fine-tuning, pseudo-label review, distillation, and
incremental-learning workflow is documented in
[`docs/campus6_learning_loop.md`](docs/campus6_learning_loop.md).

The one-command NTU120 automatic pseudo-label, hard-mining, distillation and
safe-promotion service is documented in
[`docs/ntu120_auto_evolution.md`](docs/ntu120_auto_evolution.md).

The requirement-by-requirement acceptance status and code/server verification
commands are documented in
[`docs/requirements_acceptance.md`](docs/requirements_acceptance.md). Run the
code-level acceptance suite with:

```bash
python -m dahua_cup.scripts.verify_requirements
```

NanoDet deployment, group-crop rules, and Web integration are documented in
[`docs/nanodet_person_localization.md`](docs/nanodet_person_localization.md).

The current competition gap audit, operator manual, and metric-report workflow
are documented in:

- [`docs/competition_gap_audit.md`](docs/competition_gap_audit.md)
- [`docs/user_manual.md`](docs/user_manual.md)
- [`docs/algorithm_test_guide.md`](docs/algorithm_test_guide.md)
