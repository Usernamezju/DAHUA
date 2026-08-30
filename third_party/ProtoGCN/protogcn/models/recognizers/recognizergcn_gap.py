"""ProtoGCN recognizer with a GAP-style, training-only semantic branch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..builder import RECOGNIZERS
from .recognizergcn import RecognizerGCN


@RECOGNIZERS.register_module()
class RecognizerGCNGAP(RecognizerGCN):
    """Add frozen CLIP text supervision while retaining ordinary ProtoGCN inference.

    ``semantic_text_path`` stores normalized CLIP ViT-B/32 embeddings with
    shape ``[6, 5, 512]``: global, head, arms, torso and legs.  Five learned
    projections align pooled ProtoGCN features to these fixed semantic targets.
    At evaluation this class uses the inherited classifier-only forward path.
    """

    # COCO-new indices after Kinetics_Transform adds pelvis/spine/neck.
    PARTS = (
        tuple(range(20)),                 # global
        (0, 1, 2, 3, 4),                  # head / face
        (5, 6, 7, 8, 9, 10),              # shoulders, elbows, wrists
        (5, 6, 11, 12, 17, 18, 19),       # shoulders, hips, pelvis, spine, neck
        (11, 12, 13, 14, 15, 16),          # hips, knees, ankles
    )

    def __init__(self, semantic_text_path, semantic_loss_weight=.15,
                 semantic_temperature=.07, **kwargs):
        super().__init__(**kwargs)
        text = np.load(Path(semantic_text_path)).astype(np.float32)
        if text.shape != (6, 5, 512):
            raise ValueError("semantic text embeddings must have shape [6, 5, 512], got %s" % (text.shape,))
        text = torch.from_numpy(text)
        self.register_buffer("semantic_text", F.normalize(text, dim=-1))
        self.semantic_projection = nn.ModuleList([nn.Linear(self.cls_head.in_c, 512) for _ in self.PARTS])
        self.semantic_loss_weight = float(semantic_loss_weight)
        self.semantic_temperature = float(semantic_temperature)

    def _part_feature(self, feature, joints):
        # feature: N, M, C, T, V; average people, time and the requested joints.
        return feature[..., list(joints)].mean(dim=(1, 3, 4))

    def _semantic_loss(self, feature, labels):
        losses = []
        for part_index, joints in enumerate(self.PARTS):
            skeleton = F.normalize(self.semantic_projection[part_index](self._part_feature(feature, joints)), dim=-1)
            logits = skeleton @ self.semantic_text[:, part_index, :].transpose(0, 1)
            losses.append(F.cross_entropy(logits / self.semantic_temperature, labels))
        return torch.stack(losses).mean()

    def forward_train(self, keypoint, label, **kwargs):
        assert self.with_cls_head and keypoint.shape[1] == 1
        feature, graph = self.extract_feat(keypoint[:, 0])
        labels = label.squeeze(-1)
        losses = self.cls_head.loss(self.cls_head(feature), graph, labels)
        losses["loss_gap_semantic"] = self.semantic_loss_weight * self._semantic_loss(feature, labels)
        return losses
