"""Export a Campus6 inference-only ProtoGCN checkpoint.

GAP semantic projections and CLIP text embeddings are training-only.  This
tool strips them by loading only the ordinary RecognizerGCN backbone and
six-class head, then verifies a clean forward pass of the exported artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .logits_kd import export_quantized_state_dict, load_quantized_state_dict


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", required=True)
    value.add_argument("--source-config", required=True)
    value.add_argument("--deploy-config", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    return value


def _register_protogcn(root: Path):
    import sys

    location = str(root / "gcn_models" / "ProtoGCN")
    if location not in sys.path:
        sys.path.insert(0, location)
    from protogcn.models.recognizers.recognizergcn_gap import RecognizerGCNGAP  # noqa: F401


def _model(config_path: Path, device: str, force_plain: bool):
    import mmcv
    from protogcn.models import build_recognizer

    config = mmcv.Config.fromfile(str(config_path))
    model_config = copy.deepcopy(config.model)
    if force_plain:
        model_config["type"] = "RecognizerGCN"
        model_config.pop("semantic_text_path", None)
        model_config.pop("semantic_loss_weight", None)
        model_config.pop("semantic_temperature", None)
    return build_recognizer(model_config).to(device)


def main(argv=None):
    import torch
    from mmcv.runner import load_checkpoint

    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    _register_protogcn(root)
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    source_model = _model(Path(args.source_config), args.device, force_plain=False)
    if source.suffix == ".pth":
        load_checkpoint(source_model, str(source), map_location="cpu", strict=False)
        quant_bits = None
    else:
        metadata = load_quantized_state_dict(source_model, source)
        quant_bits = int(metadata.get("quant_bits", 8))
    deploy_model = _model(Path(args.deploy_config), args.device, force_plain=True)
    source_state, deploy_state = source_model.state_dict(), deploy_model.state_dict()
    missing = [key for key in deploy_state if key not in source_state]
    if missing:
        raise RuntimeError(f"deployment state is missing {len(missing)} keys, e.g. {missing[:3]}")
    deploy_model.load_state_dict({key: source_state[key] for key in deploy_state}, strict=True)
    deploy_model.eval()
    with torch.no_grad():
        logits = deploy_model.extract_feat(torch.zeros((1, 2, 100, 20, 3), device=args.device))[0]
        logits = deploy_model.cls_head(logits)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "campus6_deployment.v1", "source": str(source), "excluded_prefixes": ["semantic_text", "semantic_projection"], "logit_shape": list(logits.shape)}
    if quant_bits is None:
        torch.save({"state_dict": deploy_model.state_dict(), "meta": metadata}, output)
    else:
        export_quantized_state_dict(deploy_model, output, metadata, num_bits=quant_bits)
    summary = {"output": str(output), "bytes": output.stat().st_size, "size_mb": output.stat().st_size / 1024.0 / 1024.0, "quant_bits": quant_bits or 32, **metadata}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
