"""Adapter around the official NanoDet PyTorch inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .grouping import PersonDetection


class NanoDetPersonDetector:
    """Load an official NanoDet checkpoint and return only person boxes."""

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        person_class_id: int | None = None,
    ) -> None:
        config_path = Path(config).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"NanoDet config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"NanoDet checkpoint not found: {checkpoint_path}"
            )
        try:
            import torch
            from nanodet.data.batch_process import stack_batch_img
            from nanodet.data.collate import naive_collate
            from nanodet.data.transform import Pipeline
            from nanodet.model.arch import build_model
            from nanodet.util import (
                Logger,
                cfg,
                load_config,
                load_model_weight,
            )
        except ImportError as exc:
            raise RuntimeError(
                "NanoDet inference requires the official nanodet package, "
                "PyTorch and OpenCV"
            ) from exc

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device is unavailable: {device}")
        load_config(cfg, str(config_path))
        logger = Logger(0, use_tensorboard=False)
        # The legacy ShuffleNetV2 constructor downloads an ImageNet backbone
        # by default.  A complete NanoDet checkpoint replaces those weights
        # immediately, so disable the redundant network dependency.
        if cfg.model.arch.backbone.name == "ShuffleNetV2":
            cfg.model.arch.backbone.update({"pretrain": False})
        model = build_model(cfg.model)
        weights = torch.load(
            str(checkpoint_path), map_location=lambda storage, _location: storage
        )
        load_model_weight(model, weights, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import (
                repvgg_det_model_convert,
            )

            model = repvgg_det_model_convert(model, deploy_model)

        class_names = [str(value) for value in cfg.class_names]
        if person_class_id is None:
            try:
                person_class_id = class_names.index("person")
            except ValueError as exc:
                raise ValueError(
                    "NanoDet config has no 'person' class; pass "
                    "--person-class-id for a custom one-class checkpoint"
                ) from exc
        if person_class_id < 0 or person_class_id >= len(class_names):
            raise ValueError("person_class_id is outside NanoDet class_names")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = resolved_device
        self.person_class_id = int(person_class_id)
        self.class_names = class_names
        self.model = model.to(resolved_device).eval()
        self.pipeline = Pipeline(
            cfg.data.val.pipeline, cfg.data.val.keep_ratio
        )
        self.input_size = cfg.data.val.input_size
        self._cfg = cfg
        self._torch = torch
        self._stack_batch_img = stack_batch_img
        self._naive_collate = naive_collate

    @staticmethod
    def parse_person_results(
        results: Mapping,
        person_class_id: int,
    ) -> list[PersonDetection]:
        """Convert NanoDet's ``class_id -> Nx5`` output to detections."""
        raw = results.get(person_class_id)
        if raw is None:
            raw = results.get(str(person_class_id), [])
        values = np.asarray(raw, dtype=np.float32)
        if values.size == 0:
            return []
        if values.ndim != 2 or values.shape[1] < 5:
            raise ValueError(
                "NanoDet person results must have Nx5 xyxy+score shape"
            )
        return [
            PersonDetection(
                bbox_xyxy=tuple(map(float, row[:4])),
                confidence=float(row[4]),
            )
            for row in values
        ]

    @staticmethod
    def select_image_results(results, image_id: int = 0) -> Mapping:
        """Unwrap one image from NanoDet's batch inference response.

        NanoDet v1.0 legacy heads return ``image_id -> class_id -> boxes``.
        Some newer wrappers return a sequence containing one class mapping per
        image, so both public response shapes are accepted here.
        """
        if isinstance(results, (list, tuple)):
            if not results:
                raise RuntimeError("NanoDet returned an empty result sequence")
            per_image = results[0]
        elif isinstance(results, Mapping):
            per_image = results.get(image_id)
            if per_image is None:
                per_image = results.get(str(image_id))
            if per_image is None and len(results) == 1:
                per_image = next(iter(results.values()))
            if per_image is None:
                raise RuntimeError(
                    "NanoDet result has no entry for image id "
                    f"{image_id}; available ids: {list(results)[:10]}"
                )
        else:
            raise RuntimeError(
                "NanoDet returned an unsupported result type: "
                f"{type(results).__name__}"
            )
        if not isinstance(per_image, Mapping):
            raise RuntimeError(
                "NanoDet per-image result must map class ids to boxes"
            )
        return per_image

    def detect(self, bgr_frame: np.ndarray) -> list[PersonDetection]:
        """Run one frame through NanoDet and return all person candidates."""
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError("NanoDet input must be an HxWx3 BGR image")
        height, width = bgr_frame.shape[:2]
        meta = {
            "img_info": {
                "id": 0,
                "file_name": None,
                "height": height,
                "width": width,
            },
            "raw_img": bgr_frame,
            "img": bgr_frame,
        }
        meta = self.pipeline(None, meta, self.input_size)
        image = self._torch.from_numpy(
            meta["img"].transpose(2, 0, 1)
        ).to(self.device)
        meta["img"] = image
        meta = self._naive_collate([meta])
        meta["img"] = self._stack_batch_img(
            meta["img"], divisible=32
        )
        # The pinned NanoDet v1.0 ``model.inference`` adds unconditional CUDA
        # synchronization even for a CPU model.  Calling the same forward and
        # post-process stages directly keeps ``--device cpu`` genuinely CPU
        # only and returns the identical detection structure.
        with self._torch.no_grad():
            predictions = self.model(meta["img"])
            results = self.model.head.post_process(predictions, meta)
        per_image = self.select_image_results(results, image_id=0)
        return self.parse_person_results(
            per_image, self.person_class_id
        )
