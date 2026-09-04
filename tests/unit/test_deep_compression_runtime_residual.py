import json
import struct

import pytest


torch = pytest.importorskip("torch")

from dahua_cup.semantic_teacher.distillation.deep_compression import (
    campus6_training_only_state_keys,
    export_huffman_runtime_residual,
)


def _package(path):
    header = {"format": "campus6.deep_compression.huffman.v1", "layers": [{"name": "backbone"}]}
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(b"DCMP1" + struct.pack("<Q", len(encoded)) + encoded)


def test_runtime_residual_omits_campus6_training_only_state(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.semantic_projection = torch.nn.Linear(2, 2)
            self.semantic_text = torch.nn.Parameter(torch.ones(2))
            self.cls_head = torch.nn.Module()
            self.cls_head.csc_loss = torch.nn.Linear(2, 2)

    model = Model()
    package = tmp_path / "weights.bin"
    sidecar = tmp_path / "residual.pth"
    _package(package)
    excluded = campus6_training_only_state_keys(model)
    report = export_huffman_runtime_residual(
        model, package, sidecar, excluded_state_keys=excluded
    )

    payload = torch.load(sidecar, map_location="cpu")
    assert report["excluded_training_only_keys"] == len(excluded)
    assert set(payload["meta"]["excluded_training_only_keys"]) == excluded
    assert "backbone.weight" not in payload["state_dict"]
    assert not excluded.intersection(payload["state_dict"])
