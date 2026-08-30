"""Shared CLI helpers: JSONL, structured logging and safe subprocesses."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path):
    source = Path(path)
    rows = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def write_jsonl(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def log_event(event, **fields):
    print(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                     ensure_ascii=False, sort_keys=True))


def require_file(path, description):
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{description} not found: {value}")
    return value


def render_command(template, **values):
    try:
        rendered = template.format_map(values)
    except KeyError as exc:
        raise ValueError(f"unknown command template field: {exc.args[0]}") from exc
    return shlex.split(rendered)


def run_command(command, dry_run=False, cwd=None, env=None):
    log_event(
        "command",
        argv=command,
        dry_run=dry_run,
        environment_overrides=sorted((env or {}).keys()),
    )
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment.update(env or {})
    return subprocess.run(
        command, cwd=cwd, env=environment, check=True
    ).returncode


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
