"""Run repository checks and optional server-readiness acceptance checks.

This command deliberately separates code-level verification from real-model
acceptance.  Passing the default checks proves that the contracts and control
flow are covered; ``--runtime`` additionally checks whether the external data,
weights and Web service required for an end-to-end run are actually present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/workspace/data/xzz_data/DAHUA")
MAXIMUM_EDGE_MODEL_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also inspect server data, model assets and the running Web API",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Web backend URL used by --runtime",
    )
    parser.add_argument(
        "--edge-model",
        action="append",
        default=[],
        help=(
            "repeat for every deployed edge model file; the 50 MiB gate "
            "is applied to their combined size"
        ),
    )
    parser.add_argument(
        "--mediapipe-model",
        default=os.environ.get("DAHUA_MEDIAPIPE_MODEL"),
        help="deployed MediaPipe task; defaults to the heavy server model",
    )
    parser.add_argument(
        "--qwen-model-dir",
        default=os.environ.get("DAHUA_QWEN_MODEL_DIR"),
        help="Qwen3-VL snapshot used by the Web teacher",
    )
    parser.add_argument(
        "--qwen-dtype",
        default=os.environ.get("DAHUA_QWEN_DTYPE", "float16"),
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument(
        "--nanodet-config",
        default=os.environ.get("DAHUA_NANODET_CONFIG"),
        help="NanoDet YAML config used by the deployed Web service",
    )
    parser.add_argument(
        "--nanodet-checkpoint",
        default=os.environ.get("DAHUA_NANODET_CHECKPOINT"),
        help="NanoDet checkpoint used by the deployed Web service",
    )
    parser.add_argument(
        "--nanodet-python",
        default=os.environ.get("DAHUA_NANODET_PYTHON", sys.executable),
        help="Python executable that imports the official nanodet package",
    )
    parser.add_argument("--output", help="optional JSON report path")
    return parser


def run_command(name: str, command: list[str]) -> Check:
    process = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout.strip()
    if len(output) > 2000:
        output = output[-2000:]
    return Check(
        name=name,
        status="pass" if process.returncode == 0 else "fail",
        detail=output or f"exit code {process.returncode}",
    )


def path_check(name: str, path: Path, kind: str) -> Check:
    present = path.is_dir() if kind == "directory" else path.is_file()
    return Check(
        name=name,
        status="pass" if present else "fail",
        detail=f"{kind}: {path}",
    )


def video_check(root: Path) -> Check:
    extensions = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
    count = 0
    if root.is_dir():
        count = sum(
            1
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
    return Check(
        name="candidate_videos",
        status="pass" if count else "fail",
        detail=f"{count} video(s) below {root}",
    )


def url_json_check(name: str, url: str) -> Check:
    try:
        with urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return Check(name, "pass", json.dumps(payload, ensure_ascii=False)[:1000])
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        return Check(name, "fail", f"{url}: {exc}")


def edge_model_check(paths: list[str]) -> Check:
    if not paths:
        return Check(
            "edge_model_bundle_size",
            "skip",
            "repeat --edge-model for every file in the deployable bundle",
        )
    sources = [Path(path).expanduser().resolve() for path in paths]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        return Check(
            "edge_model_bundle_size",
            "fail",
            "missing file(s): " + ", ".join(missing),
        )
    size = sum(path.stat().st_size for path in sources)
    return Check(
        "edge_model_bundle_size",
        "pass" if size <= MAXIMUM_EDGE_MODEL_BYTES else "fail",
        (
            f"{len(sources)} file(s), {size} bytes "
            f"({size / 1024 / 1024:.2f} MiB), "
            f"limit {MAXIMUM_EDGE_MODEL_BYTES} bytes"
        ),
    )


def qwen_profile_check(path: Path, dtype: str) -> Check:
    if not path.is_dir():
        return Check(
            "qwen_runtime_profile", "fail", f"directory not found: {path}"
        )
    config = path / "config.json"
    if not config.is_file():
        return Check(
            "qwen_runtime_profile", "fail", f"config.json not found: {path}"
        )
    try:
        value = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return Check("qwen_runtime_profile", "fail", str(exc))
    architectures = list(value.get("architectures") or [])
    compatible = "Qwen3VLForConditionalGeneration" in architectures
    return Check(
        "qwen_runtime_profile",
        "pass" if compatible else "fail",
        f"model={path.name}, dtype={dtype}, architectures={architectures}",
    )


def optional_file_check(name: str, value: str | None) -> Check:
    if not value:
        return Check(name, "skip", f"provide --{name.replace('_', '-')}")
    return path_check(name, Path(value).expanduser().resolve(), "file")


def nanodet_import_check(python_bin: str, configured: bool) -> Check:
    if not configured:
        return Check(
            "nanodet_runtime_import",
            "skip",
            "NanoDet config/checkpoint were not both provided",
        )
    return run_command(
        "nanodet_runtime_import",
        [
            python_bin,
            "-c",
            "import cv2, nanodet, torch; print('NanoDet runtime imports OK')",
        ],
    )


def code_checks() -> list[Check]:
    if importlib.util.find_spec("pytest") is None:
        test_check = Check(
            "pytest",
            "fail",
            (
                "pytest is not installed; run "
                f"{sys.executable} -m pip install -r "
                "dahua_cup/requirements-test.txt"
            ),
        )
    else:
        test_check = run_command(
            "pytest",
            [sys.executable, "-m", "pytest", "-q", "dahua_cup/tests"],
        )
    checks = [
        test_check,
        run_command(
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "dahua_cup"],
        ),
        run_command(
            "visualization_shell_syntax",
            ["bash", "-n", "dahua_cup/scripts/run_visualization.sh"],
        ),
        run_command(
            "stn_visualization_shell_syntax",
            [
                "bash",
                "-n",
                "dahua_cup/scripts/run_stn_visualization.sh",
            ],
        ),
    ]
    if shutil.which("node"):
        checks.extend(
            [
                run_command(
                    "visualization_javascript_syntax",
                    ["node", "--check", "dahua_cup/visualization/app.js"],
                ),
                run_command(
                    "stn_visualization_javascript_syntax",
                    ["node", "--check", "dahua_cup/stn/web/app.js"],
                ),
            ]
        )
    else:
        checks.extend(
            [
                Check(
                    "visualization_javascript_syntax",
                    "skip",
                    "node is not installed",
                ),
                Check(
                    "stn_visualization_javascript_syntax",
                    "skip",
                    "node is not installed",
                ),
            ]
        )

    modules = (
        "dahua_cup.pipeline.nanodet_group_crop_worker",
        "dahua_cup.pipeline.mediapipe_pose_worker",
        "dahua_cup.pipeline.protogcn_student_worker",
        "dahua_cup.pipeline.qwen_teacher_worker",
        "dahua_cup.pipeline.mine_candidates",
        "dahua_cup.pipeline.collect_ntu120_pseudo",
        "dahua_cup.pipeline.build_ntu120_distill_dataset",
        "dahua_cup.pipeline.train_ntu120_distill",
        "dahua_cup.pipeline.export_weight_checkpoint",
        "dahua_cup.pipeline.evaluate_predictions",
        "dahua_cup.pipeline.generate_pseudo_dataset",
        "dahua_cup.pipeline.build_campus6_dataset",
        "dahua_cup.pipeline.train_campus6",
        "dahua_cup.pipeline.build_replay_manifest",
        "dahua_cup.pipeline.register_candidate",
        "dahua_cup.pipeline.validate_candidate",
        "dahua_cup.pipeline.rollback_model",
        "dahua_cup.stn.export_yolo_teacher",
        "dahua_cup.stn.train",
        "dahua_cup.stn.visualization_app",
    )
    for module in modules:
        checks.append(
            run_command(
                f"cli:{module.rsplit('.', 1)[-1]}",
                [sys.executable, "-m", module, "--help"],
            )
        )
    return checks


def runtime_checks(
    data_root: Path,
    base_url: str,
    edge_models: list[str],
    mediapipe_model: str | None,
    qwen_model_dir: str | None,
    qwen_dtype: str,
    nanodet_config: str | None,
    nanodet_checkpoint: str | None,
    nanodet_python: str,
) -> list[Check]:
    model_root = data_root / "models"
    nanodet_configured = bool(nanodet_config and nanodet_checkpoint)
    pose_model = (
        Path(mediapipe_model).expanduser().resolve()
        if mediapipe_model
        else model_root
        / "pose_models"
        / "mediapipe"
        / "pose_landmarker_heavy.task"
    )
    public_qwen = Path(
        "/workspace/data/public_data/Qwen3-VL-8B-Instruct"
    )
    qwen_model = (
        Path(qwen_model_dir).expanduser().resolve()
        if qwen_model_dir
        else public_qwen
        if public_qwen.is_dir()
        else model_root / "Qwen3-VL-8B-Instruct"
    )
    return [
        path_check(
            "mediapipe_model",
            pose_model,
            "file",
        ),
        path_check(
            "ntu120_student_checkpoint",
            model_root
            / "protogcn_pretrained"
            / "ntu120_xsub_bone_b1"
            / "best_top1_acc_epoch_150.pth",
            "file",
        ),
        qwen_profile_check(qwen_model, qwen_dtype),
        path_check(
            "ntu120_base_annotation",
            data_root
            / "datasets"
            / "NTU"
            / "ProtoGCN"
            / "ntu120_3danno.pkl",
            "file",
        ),
        optional_file_check("nanodet_config", nanodet_config),
        optional_file_check("nanodet_checkpoint", nanodet_checkpoint),
        nanodet_import_check(nanodet_python, nanodet_configured),
        video_check(data_root / "datasets" / "vedio"),
        url_json_check("web_health", f"{base_url.rstrip('/')}/api/health"),
        url_json_check(
            "web_capabilities",
            f"{base_url.rstrip('/')}/api/capabilities",
        ),
        url_json_check("web_system", f"{base_url.rstrip('/')}/api/system"),
        url_json_check(
            "auto_evolution",
            f"{base_url.rstrip('/')}/api/evolution",
        ),
        edge_model_check(edge_models),
    ]


def print_report(checks: list[Check]) -> None:
    labels = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}
    for check in checks:
        print(f"[{labels[check.status]}] {check.name}: {check.detail}")
    passed = sum(check.status == "pass" for check in checks)
    failed = sum(check.status == "fail" for check in checks)
    skipped = sum(check.status == "skip" for check in checks)
    print(f"\nsummary: {passed} passed, {failed} failed, {skipped} skipped")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    checks = code_checks()
    if args.runtime:
        checks.extend(
            runtime_checks(
                Path(args.data_root).expanduser().resolve(),
                args.base_url,
                args.edge_model,
                args.mediapipe_model,
                args.qwen_model_dir,
                args.qwen_dtype,
                args.nanodet_config,
                args.nanodet_checkpoint,
                args.nanodet_python,
            )
        )
    print_report(checks)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "schema_version": "requirements_verification.v1",
                    "runtime_checked": bool(args.runtime),
                    "checks": [asdict(check) for check in checks],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 1 if any(check.status == "fail" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
