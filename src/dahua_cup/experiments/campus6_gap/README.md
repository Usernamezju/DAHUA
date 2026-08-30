# Campus6 Skeleton Experiment Matrix

This directory deliberately contains **no data path or checkpoint path**.
Render a runnable config only after the dataset split and pre-trained weight
have been approved:

```bash
python scripts/render_config.py configs/ctr_gcn_full.yaml.in --output <generated-config.yaml> \
  --set CAMPUS6_CTR_DATA=<npz> --set CAMPUS6_CTR_WORK_DIR=<work-dir> \
  --set CAMPUS6_CTR_PRETRAINED=<checkpoint>
```

## Experiments

| ID | Backbone | Training objective | Output used for later ensemble |
|---|---|---|---|
| `ctr_full` | CTR-GCN | Cross entropy, all layers trainable | best validation checkpoint |
| `proto_full` | ProtoGCN | Cross entropy, all layers trainable | best validation checkpoint |
| `gap_full` | CTR-GCN + GAP four-part heads | CE + GAP skeleton/text alignment | best validation checkpoint |
| `*_int8` | Corresponding ONNX model | static QDQ INT8 calibration; no retraining | individual INT8 ONNX model |
| `int8_ensemble` | all available INT8 models | mean calibrated probabilities | every pair and the three-way ensemble |

The `gap_full` control differs from `ctr_full` only by GAP's input transform,
four part projections and semantic alignment loss. It remains a full
fine-tune: no backbone layers are frozen.

## Input contracts

* CTR-GCN / GAP use the repository's standard feeder `.npz` contract.
* ProtoGCN uses its regular annotation file configured through its rendered
  Python template.
* Quantization and ensemble consume a framework-neutral `.npz` with:
  `data` (`N,C,T,V,M` float32), `label` (`N` int64), and optional `sample_id`.
* All models must use the same six-class order from the prompt banks.  Use
  `prompts/campus6_gap_prompt_bank_en.yaml` for the executable GAP run: the
  official OpenAI CLIP text encoder is English-centric.  The corresponding
  Chinese version is included only for human prompt review.

## INT8 protocol

Export each selected FP32 model to ONNX with a dynamic batch axis. Calibrate
only on training samples, then evaluate the FP32 and INT8 ONNX exports on the
unchanged validation/test split. Do not average quantized tensors: each INT8
model runs independently and the ensemble averages its six-class softmax
probabilities. This is the valid way to ensemble quantized model weights.

## Decision rule

Report one row for each FP32 model, each INT8 model, all three INT8 pairs and
the three-way INT8 ensemble. Each row must include top-1, macro-F1 and
per-class recall; select a deployment model only using validation results.
