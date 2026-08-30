# Semantic teacher

The public backend is
`dahua_cup.semantic_teacher.api.qwen_api.QwenTeacher`. Construction
is lazy and local-only: importing it does not select a GPU or load weights. Pass
an explicit `Qwen3Config`, or set `DAHUA_MODEL_ROOT` for the compatibility
constructor.

Teacher outputs are validated against `teacher_output.v1`; pseudo-label
production uses `dahua_cup.pipeline.generate_pseudo_dataset`, which combines repeated
teacher distributions, student agreement, pose quality, temporal stability and
measured-evidence conflict rules. Review items are stored in the same SQLite
database used by the Web application through `dahua_cup.backend.store.ReviewStore`.
Submitting or undoing a Web review atomically rewrites the corresponding
versioned pseudo-label JSONL record.

Teacher probabilities can be temperature-calibrated against an independent
human-labelled calibration split with `dahua_cup.pipeline.calibrate_teacher`.
This calibrates the teacher's six-way distribution; it does not reinterpret a
freely written confidence number as an observed accuracy.
