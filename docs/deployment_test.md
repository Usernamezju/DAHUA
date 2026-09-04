# From-scratch deployment test

Goal: on a clean Linux machine, download the project and run the full loop
"video upload → skeleton extraction → student inference → hard cases" to
verify the project works. All commands below are CPU-only.

## 1. Clone

```bash
git clone git@github.com:Usernamezju/DAHUA.git
cd DAHUA
```

## 2. Download model weights (GitHub release asset)

```bash
bash scripts/download_models.sh
```

Downloads `models.tar.gz`, verifies its SHA-256 against the published
`models.tar.gz.sha256` sidecar, unpacks it into the repository root, and
checks that `models/MANIFEST.json` is in place. No authentication needed.

## 3. Create the two role environments

Two environments are required because the delivered ProtoGCN student model
pins MMCV 1.5.0 while RTMPose extraction needs MMCV 2.1.0.

### 3.1 Web + pose extraction (`dahua_test_web`, Python 3.10)

```bash
conda create -n dahua_test_web python=3.10 -y
conda activate dahua_test_web

python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

python -m pip install \
  mmengine==0.10.5 mmdet==3.3.0 mmpose==1.3.2 \
  opencv-python fastapi uvicorn python-multipart
```

### 3.2 Student inference (`dahua_test_gcn`, Python 3.8)

```bash
conda create -n dahua_test_gcn python=3.8 -y
conda activate dahua_test_gcn

python -m pip install torch==1.10.2+cpu torchvision==0.11.3+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install -r requirements/skel.txt \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

python -m pip install python-multipart
```

## 4. Launch (CPU full loop)

```bash
# CPU test uses the MMPose FP32 diagnostic backend; override the default
# TensorRT FP16 backend, which requires a GPU.
export DAHUA_POSE_BACKEND=mmpose

bash scripts/run_local_visualization.sh \
  --web-python ~/miniconda3/envs/dahua_test_web/bin/python \
  --rtmpose-python ~/miniconda3/envs/dahua_test_web/bin/python \
  --student-python ~/miniconda3/envs/dahua_test_gcn/bin/python \
  --device cpu \
  2>&1 | tee runtime/local_web.log
```

The launcher preflights the three environments; on success it prints
`Starting local Campus6 Web at http://127.0.0.1:8000`.

## 5. Verification checklist

1. Open `http://127.0.0.1:8000`; the seven navigation pages respond.
2. Upload an RGB video on the "GCN 推理可视化" page.
3. The job log shows `rtmpose17_pose_complete` (with `elapsed_seconds`) and
   `campus6_rtmpose17_inference_complete` (with `model_load_seconds`,
   `inference_seconds`, `total_seconds`).
4. The result page shows six-class Top-K probabilities and the skeleton video.
5. Optional: configure Qwen SSH/API in Web settings for teacher arbitration.
6. Optional: place `annotations_with_all.pkl` under `dataset/campus6_baseline/`
   to browse the 358-sample baseline.

## 6. GPU / TensorRT FP16 (optional)

```bash
conda create -n dahua_test_pose_fp16 python=3.8 -y
conda activate dahua_test_pose_fp16
python -m pip install -r requirements/pose_fp16.txt

bash scripts/run_local_visualization.sh \
  --web-python ~/miniconda3/envs/dahua_test_web/bin/python \
  --rtmpose-python ~/miniconda3/envs/dahua_test_pose_fp16/bin/python \
  --student-python ~/miniconda3/envs/dahua_test_gcn/bin/python \
  --device gpu
```

## Publishing the release asset (for maintainers)

```bash
gh auth login
gh release create v1.0.0-rc1 \
  models.tar.gz models.tar.gz.sha256 \
  --title "Campus6 model assets v1.0.0-rc1" \
  --notes "M1KD INT8 student, FP32 pose weights, TensorRT FP16 engines, pretrained checkpoints"
```
