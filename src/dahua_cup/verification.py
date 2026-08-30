"""Verify the self-contained Campus6 code and optional runtime assets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from dahua_cup.paths import CONFIG_ROOT, PROTOGCN_ROOT, REPOSITORY_ROOT


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--runtime", action="store_true", help="also inspect local model files")
    value.add_argument("--checkpoint", default=str(REPOSITORY_ROOT / "models/student/campus6_protogcn_gap_fp32_epoch40.pth"))
    value.add_argument("--int8-checkpoint", default=str(REPOSITORY_ROOT / "models/student/M1KD.int8.pt"))
    value.add_argument("--output", help="optional JSON report path")
    return value


def path_check(name: str, path: Path) -> Check:
    return Check(name, "pass" if path.is_file() else "fail", str(path))


def command_check(name: str, command: list[str]) -> Check:
    process = subprocess.run(command, cwd=REPOSITORY_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = process.stdout.strip()[-2000:]
    return Check(name, "pass" if process.returncode == 0 else "fail", output or f"exit code {process.returncode}")


def code_checks() -> list[Check]:
    checks = [
        path_check("campus6_labels", CONFIG_ROOT / "campus/campus6_labels.txt"),
        path_check("protogcn_campus6_config", PROTOGCN_ROOT / "configs/campus6/rtmpose26_k400_2d_gap_full.py"),
        Check("pytest_available", "pass" if importlib.util.find_spec("pytest") else "skip", "pytest optional dependency"),
    ]
    if checks[-1].status == "pass":
        checks.append(command_check("unit_tests", [sys.executable, "-m", "pytest", "-q", "tests/unit"]))
    return checks


def runtime_checks(args: argparse.Namespace) -> list[Check]:
    return [
        path_check("fp32_checkpoint", Path(args.checkpoint).expanduser().resolve()),
        path_check("int8_deployment_checkpoint", Path(args.int8_checkpoint).expanduser().resolve()),
    ]


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    checks = code_checks() + (runtime_checks(args) if args.runtime else [])
    result = {"schema_version": "campus6_verification.v1", "checks": [asdict(item) for item in checks]}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    return 1 if any(item.status == "fail" for item in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
