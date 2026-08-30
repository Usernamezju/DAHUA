# Campus6 multimodal candidate plan

## Capacity contract

For each Campus6 class, the candidate pool must contain at least 100 RGB videos
and at least 500 RGB plus skeleton sequences. Candidate retrieval hints are not
ground-truth labels. Final labels are assigned independently by the multimodal
screening workflow and stored separately by modality.

## Current server capacity

| Target class | RGB | Skeleton | Combined | Status |
|---|---:|---:|---:|---|
| `normal_walk` | 136 | 2,914 | 3,050 | ready |
| `normal_run` | 211 | 1,589 | 1,800 | ready |
| `playful_chase` | 16 | 960 | 976 | needs 84 RGB |
| `playful_push` | 20 | 1,333 | 1,353 | needs 80 RGB |
| `conflict_chase` | 5 | 960 | 965 | needs 95 RGB |
| `conflict_push` | 52 | 2,061 | 2,113 | needs 48 RGB |

The current combined quota is met in all six classes. Supplemental download is
therefore restricted to interaction RGB candidates.

## Source allocation

| Modality | Source | Campus6 retrieval use |
|---|---|---|
| RGB | KTH | normal walk/run |
| RGB | BEHAVE | two-person walk/run/chase/fight candidates |
| RGB | LIMU | push/pull/punch/kick interaction candidates |
| RGB | Kinetics-700 | pillow fight, wrestling, slapping, person punching |
| RGB | Creative Commons search | playful/conflict chase and push shortfalls |
| Skeleton | NTU A059/A060/A099 | walk/run candidates |
| Skeleton | NTU A116 | playful/conflict chase candidates |
| Skeleton | NTU A052 | playful/conflict push candidates |
| Skeleton | Kinetics Skeleton | walk/run/wrestling/slapping/punching candidates |

HMDB51 `walk`, `run`, and `push` are optional capacity reserves. Raw NTU RGB is
not required: the installed `.skeleton` files are rendered into pose-only
temporal sheets directly.

## Bias and leakage controls

- Filenames, source dataset names, source labels, and retrieval hints are never
  included in model prompts.
- A candidate may carry several retrieval hints, but a screened sample receives
  zero or one final Campus6 class.
- Skeleton interaction labels require three-model unanimity.
- Crowded or incomplete clips are retained for manual review rather than forced
  into a quota class.
- RGB and skeleton labels have separate manifests and storage trees.
- RGB and skeleton candidates must not share an original source-video identity;
  Kinetics/YouTube IDs are excluded during download and audited again during
  manifest construction.
- Quota checks report shortfalls and never overwrite a model decision.

## Completion evidence

Candidate acquisition is complete only when
`candidate_summary.json` reports `ready: true` for all classes. Dataset
screening is complete only when `campus6_screened/manifests/summary.json`
reports both zero RGB shortfall and zero combined shortfall for all classes.
