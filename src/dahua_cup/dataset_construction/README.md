# Campus6 multimodal dataset construction

This directory builds and screens a multimodal candidate pool for the six
Campus6 classes. RGB videos and skeleton sequences share one screening policy,
but their accepted labels and files are stored separately. The workflow uses
hosted vision-language APIs and CPU preprocessing; it does not run a local LLM,
an Agent, or a server GPU.

## Quotas and current server inventory

The candidate-pool requirements are checked independently for every class:

- at least 100 RGB videos;
- at least 500 RGB plus skeleton candidates.

The latest server audit produced 7,928 unique candidates:

| Campus6 class | RGB | Skeleton | Combined | RGB shortfall |
|---|---:|---:|---:|---:|
| `normal_walk` | 136 | 2,914 | 3,050 | 0 |
| `normal_run` | 211 | 1,589 | 1,800 | 0 |
| `playful_chase` | 16 | 960 | 976 | 84 |
| `playful_push` | 20 | 1,333 | 1,353 | 80 |
| `conflict_chase` | 5 | 960 | 965 | 95 |
| `conflict_push` | 52 | 2,061 | 2,113 | 48 |

The combined quota is already satisfied for every class. The downloader only
needs to fill the four interaction-class RGB shortfalls. Counts are candidate
retrieval capacity, not accepted ground truth: one ambiguous interaction may be
a candidate for both playful and conflict intent, but it can receive at most one
final Campus6 label after blind screening.

## Server paths

| Purpose | Path |
|---|---|
| Repository | `/workspace/code/DAHUA` |
| Existing RGB | `/workspace/data/xzz_data/DAHUA/datasets/vedio` |
| NTU skeleton | `/workspace/data/xzz_data/DAHUA/datasets/NTU/NTU_skeleton` |
| Kinetics skeleton | `/workspace/data/xzz_data/DAHUA/datasets/kinetics-skeleton/processed-1` |
| Supplemental downloads | `/workspace/data/xzz_data/DAHUA/datasets/campus6_downloads` |
| Candidate manifests | `/workspace/data/xzz_data/DAHUA/datasets/campus6_candidates/manifests` |
| Screened results | `/workspace/data/xzz_data/DAHUA/datasets/campus6_screened` |
| Logs | `/workspace/data/xzz_data/DAHUA/runtime/logs/dataset_construction` |
| API keys | `/workspace/code/DAHUA/.secrets/campus6_screening.env` |

All paths can be overridden by CLI arguments or `DAHUA_*` environment
variables. `.secrets/` and `.tools/` are ignored by Git.

## Candidate sources

- Existing RGB: KTH, BEHAVE, and LIMU.
- Existing skeleton: NTU RGB+D 120 and Kinetics Skeleton.
- Supplemental RGB: official Kinetics-700 annotations plus Creative Commons
  search candidates. HMDB51 `walk`, `run`, and `push` are reused when `7z` is
  available, but they are not required for the current normal-class floor.

The manifest builder records source, source label, modality, licensing notes,
and retrieval hints. The source label and filename are deliberately excluded
from all model prompts.

RGB and skeleton candidates are source-disjoint. Every Kinetics skeleton
YouTube ID is excluded before RGB download, and the manifest builder repeats
the identity audit as a hard gate. If an RGB file and a skeleton sequence are
derived from the same original video, the RGB record is excluded and the
downloader continues searching for a different video.

## One-click candidate download

First inspect the planned sources without downloading:

```bash
cd /workspace/code/DAHUA

bash dahua_cup/dataset_construction/download_multimodal_candidates.sh \
  --dry-run
```

Then run the resumable downloader and rebuild the quota-checked manifest:

```bash
cd /workspace/code/DAHUA

bash dahua_cup/dataset_construction/download_multimodal_candidates.sh \
  2>&1 | tee \
  /workspace/data/xzz_data/DAHUA/runtime/logs/dataset_construction/download_multimodal_candidates.log
```

The wrapper installs a repository-local standalone `yt-dlp`, uses the existing
Conda `ffmpeg`, resumes existing files, and fails honestly if the post-download
manifest still misses a quota. It does not use a GPU. Unavailable or deleted
web videos are logged as shortfalls and may be supplied by self-collected clips.

To rebuild the manifest without downloading:

```bash
cd /workspace/code/DAHUA

/root/miniconda3/envs/skel_gcn38/bin/python -m \
  dahua_cup.dataset_construction.build_candidate_manifest \
  --config \
  /workspace/code/DAHUA/dahua_cup/dataset_construction/configs/multimodal_candidates.json \
  --require-quotas
```

## Screening policy

RGB videos are converted to time-ordered contact sheets. Skeleton sequences are
rendered directly as pose-only temporal sheets, so NTU RGB is not required.

1. `glm-4.6v-flash` performs a blind primary review.
2. `glm-4.1v-thinking-flash` performs a blind counter-review.
3. If they disagree and `DASHSCOPE_API_KEY` is available, `qwen3-vl-plus`
   independently adjudicates the sample.
4. RGB accepts a full class after two valid reviews agree.
5. Skeleton interaction classes require unanimous agreement from all three
   models because pose-only input cannot reliably expose facial expression,
   scene context, or intent.
6. Disagreement, crowd ambiguity, incomplete action, or uncertain intent is
   retained for manual review or as a partial `chase`/`push` motion label.

No free-form numerical confidence is treated as calibrated probability. The
decision is based on validated structured outputs and cross-model agreement.

## One-time API setup

Install the lightweight CPU dependencies in the existing environment:

```bash
cd /workspace/code/DAHUA

/root/miniconda3/envs/skel_gcn38/bin/python -m pip install \
  -r dahua_cup/dataset_construction/requirements-screening.txt
```

Create the ignored credential file once:

```bash
cd /workspace/code/DAHUA

mkdir -p /workspace/code/DAHUA/.secrets

cp \
  dahua_cup/dataset_construction/credentials.env.example \
  /workspace/code/DAHUA/.secrets/campus6_screening.env

chmod 600 \
  /workspace/code/DAHUA/.secrets/campus6_screening.env
```

Set `ZHIPUAI_API_KEY`; `DASHSCOPE_API_KEY` is optional but required for the
third-model adjudication and skeleton interaction acceptance.

## One-click multimodal screening

Use a dry run first. This rebuilds and audits the manifest without API calls:

```bash
cd /workspace/code/DAHUA

bash dahua_cup/dataset_construction/run_multimodal_screening.sh \
  --dry-run
```

Run the resumable hosted-model screening:

```bash
cd /workspace/code/DAHUA

bash dahua_cup/dataset_construction/run_multimodal_screening.sh \
  2>&1 | tee \
  /workspace/data/xzz_data/DAHUA/runtime/logs/dataset_construction/screen_multimodal_candidates.log
```

Completed records whose config, prompts, and source fingerprint have not
changed are resumed without another API request. RGB candidates are scheduled
before skeleton candidates within each retrieval class so the RGB floor is not
starved. Screening stops only after every accepted class has at least 100 RGB
and 500 combined labeled samples, or after all candidates are exhausted.

Useful bounded checks:

```bash
# Generate evidence sheets only; do not call an API.
bash dahua_cup/dataset_construction/run_multimodal_screening.sh \
  --prepare-only \
  --limit 20

# End-to-end API smoke test on one candidate.
bash dahua_cup/dataset_construction/run_multimodal_screening.sh \
  --limit 1
```

## Outputs

- `labeled/rgb/<label>/`: accepted RGB video links.
- `labeled/skeleton/<label>/`: accepted skeleton file links.
- `partial_labels/rgb/...` and `partial_labels/skeleton/...`: chase/push motion
  with unresolved intent.
- `manual_review/rgb/...` and `manual_review/skeleton/...`: unresolved samples.
- `reviews/*.json`: API responses, validation errors, evidence hashes, and final
  decisions.
- `work/contact_sheets/`: RGB or pose-only evidence sent to the models.
- `manifests/labeled_rgb.jsonl`: authoritative accepted RGB labels.
- `manifests/labeled_skeleton.jsonl`: authoritative accepted skeleton labels.
- `manifests/final_dataset.jsonl`: combined auditable result.
- `manifests/summary.json`: RGB, skeleton, combined counts and both quota types.

Hard links avoid duplicating source storage. If source and output are on
different filesystems, the workflow falls back to symbolic links.
