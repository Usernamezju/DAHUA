# Campus6 TensorRT FP16 pose package

This directory contains the validated deployment package for the direct
COCO-17 Campus6 skeleton extractor:

- `rtmdet_nano_person/end2end.engine` — RTMDet-nano-person, FP16
- `rtmpose_s/end2end.engine` — RTMPose-S, COCO-17, FP16
- MMDeploy `deploy.json`, `detail.json`, and `pipeline.json` metadata
- `validation.json` — quality and runtime validation record

The engine SHA-256 values are:

```text
c99df29de40ffc235931aea85634d8b35fedccab6a875a1b5cd9f70cce9dab84  rtmdet_nano_person/end2end.engine
a7c29c23340dc8c10e8dc01c89e3694b4a21d2ffae231d0c7792c4f0bff12957  rtmpose_s/end2end.engine
```

Use `requirements/pose_fp16.txt` and source
`configs/pose/rtmpose_fp16.env.example` before starting the worker. The
reference deployment selects `rtmpose17`, `tensorrt_fp16`, and `cuda:0`.
