"""Emit COCO annotations for RTMPose-S INT8 calibration.

Runs RTMDet over calibration frames and writes ``annotations.json`` with
person boxes, so the pose calibration loader's TopDownAffine crops are valid.
An approved INT8 detector can be used directly.  During PTQ investigation a
trusted FP32 detector is also supported as an *offline calibration-only*
source; it never changes the deployment backend.

Usage (on the export host):

    DAHUA_POSE_INT8_CALIB_DIR=/path/to/frames \
    LD_LIBRARY_PATH=<tensorrt_libs> \
    <int8-env>/bin/python configs/pose/generate_pose_calib_bboxes.py \
      --detector-model-dir models/pose/int8/rtmdet_s \
      --score 0.30 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--detector-model-dir")
    value.add_argument("--detector-config")
    value.add_argument("--detector-checkpoint")
    value.add_argument("--score", type=float, default=0.30)
    value.add_argument("--device", default="cuda:0")
    return value


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    calib_dir = Path(os.environ["DAHUA_POSE_INT8_CALIB_DIR"])
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"calibration dir not found: {calib_dir}")
    images = sorted(calib_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"no .jpg frames in {calib_dir}")

    import cv2

    use_fp32 = bool(args.detector_config or args.detector_checkpoint)
    if use_fp32:
        if not (args.detector_config and args.detector_checkpoint):
            raise ValueError(
                "--detector-config and --detector-checkpoint must be supplied together"
            )
        from mmdet.apis import inference_detector, init_detector

        detector = init_detector(
            args.detector_config, args.detector_checkpoint, device=args.device
        )

        def detect(image):
            result = inference_detector(detector, image)
            instances = result.pred_instances
            return (
                instances.bboxes.detach().cpu().numpy(),
                instances.scores.detach().cpu().numpy(),
                instances.labels.detach().cpu().numpy(),
            )
    else:
        if not args.detector_model_dir:
            raise ValueError(
                "provide --detector-model-dir or FP32 --detector-config/--detector-checkpoint"
            )
        kind, _, device_id = args.device.partition(":")
        from mmdeploy_runtime import Detector

        detector = Detector(
            args.detector_model_dir,
            device_name=kind or "cuda",
            device_id=int(device_id or 0),
        )

        def detect(image):
            result = detector(image)
            boxes = np.asarray(result[0], dtype=np.float32).reshape(-1, 5)
            return boxes[:, :4], boxes[:, 4], np.asarray(result[1]).reshape(-1)

    annotations = []
    image_metadata = []
    annotation_id = 1
    for image_id, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"unable to read calibration frame: {image_path}")
        height, width = image.shape[:2]
        image_metadata.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": int(width),
            "height": int(height),
        })
        bboxes, scores, labels = detect(image)
        for box, score, label in zip(bboxes, scores, labels):
            if int(label) != 0 or box.shape[0] != 4:
                continue
            x1, y1, x2, y2 = box
            width, height = float(x2 - x1), float(y2 - y1)
            if float(score) < args.score or width <= 1 or height <= 1:
                continue
            # RTMPose's validation pipeline uses only ``bbox`` to make the
            # TopDownAffine crop.  It nevertheless filters a COCO instance
            # whose ``num_keypoints`` is zero before that pipeline runs.
            # These centre placeholders are therefore dataset plumbing only:
            # they are never used as targets or model inputs during PTQ.
            centre_keypoint = [float(x1 + width / 2), float(y1 + height / 2), 2]
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), width, height],
                "area": width * height,
                "iscrowd": 0,
                "keypoints": centre_keypoint * 17,
                "num_keypoints": 17,
                "score": float(score),
            })
            annotation_id += 1

    payload = {
        "images": image_metadata,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    output = calib_dir / "annotations.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    annotated = len({ann["image_id"] for ann in annotations})
    print(
        f"frames={len(images)} annotations={len(annotations)} "
        f"frames_with_person={annotated}"
    )


if __name__ == "__main__":
    main()
