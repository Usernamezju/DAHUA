# Prompt audit

The prompts were reviewed against the Campus6 data-construction goal.

## Passed checks

- **Label boundaries:** walking/running are separated from targeted pursuit;
  chase is not inferred from two people merely running together; contact alone
  is not sufficient for pushing.
- **Playful versus conflict:** the prompt requires temporal evidence such as role
  exchange, return, escape, defensive posture, displacement or loss of balance.
- **Person-count constraint:** all four interaction labels require exactly two
  primary people and reject a persistent third participant as `crowded`.
- **Evidence completeness:** full labels require the action beginning, key event
  and outcome, good/acceptable visibility and evidence grade A.
- **No fake confidence:** models output evidence grade A/B/C, not an arbitrary
  decimal confidence. Acceptance is produced by model agreement plus schema
  checks in code.
- **Blind review:** neither reviewer receives filenames, source labels, dataset
  names, the other reviewer's answer, nor current class quotas.
- **Conservative fallback:** ambiguous intent becomes a partial motion label;
  disagreement becomes Qwen adjudication or manual review.
- **Machine validation:** output is strict JSON and all enums, booleans, person
  counts, label-motion-intent consistency and accept/reject rules are validated.

## Remaining human responsibility

Hosted model agreement is a pseudo-label, not ground truth. Before publishing a
dataset, manually audit at least a stratified random 10% per class and all model
release errors. Report precision per class and return failing patterns to the
prompt/rule set; any prompt change automatically invalidates the resume signature.
