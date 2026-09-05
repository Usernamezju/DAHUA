"""Package the two Campus6 model release assets from the authoritative MANIFEST.

The release no longer ships a single ``models.tar.gz``.  Instead two assets are
published, each consumed by exactly one deployment side:

- ``models-server.tar.gz`` — the FP32 GCN training/regression baseline and the
  GAP semantic embeddings for the server (training + Qwen host).
- ``models-edge.tar.gz`` — the TensorRT FP16 pose engines under
  ``models/pose/fp16/`` and the M1KD INT8 student checkpoint for local
  inference.

``models/MANIFEST.json`` is the single authoritative manifest.  Every entry
carries a ``release_asset`` field naming its package (``null`` entries are not
packaged).  In addition, the edge package always collects
``models/pose/fp16/**`` recursively and the ``M1KD.int8.pt.json`` sidecar
sharing the student checkpoint's stem.  Tar paths keep the ``models/`` prefix
so archives unpack into the repository root, matching the previous
``models.tar.gz`` behaviour.

Each archive is published with:

- a ``<archive>.sha256`` sidecar holding the whole-archive digest, and
- in-archive ``models/MANIFEST.<server|edge>.json`` and
  ``models/SHA256SUMS.<server|edge>`` files for per-file verification.

Build products are written under the output directory (``dist/`` by default,
already git-ignored) so ``models/MANIFEST.*.json`` never pollutes the working
tree, which would otherwise happen because of the ``!models/**/*.json``
gitignore exception.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

# Packaging is deterministic: fixed in-archive mtime and gzip header time so
# rebuilding the same tree reproduces the same archive SHA-256.
FIXED_MTIME = 0

SIDECAR_RULES = {
    "models/student/M1KD.int8.pt": ("models/student/M1KD.int8.pt.json",),
}

EDGE_EXTRA_ROOTS = ("models/pose/fp16",)

# SHA-256 pairs documented in models/pose/fp16/README.md; the packager refuses
# to ship engines that do not match the published record.
FP16_README = "models/pose/fp16/README.md"
DIGEST_LINE = re.compile(r"^([0-9a-f]{64})\s+(\S+)\s*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=None,
                        help="Repository root containing models/ (default: script's repository)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to models/MANIFEST.json (default: <models-root>/models/MANIFEST.json)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for archives and sidecars (default: <repo>/dist)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the package plan without writing anything")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    repo_root = script.parents[1]
    models_root = (args.models_root or repo_root).resolve()
    manifest = (args.manifest or models_root / "models" / "MANIFEST.json").resolve()
    output_dir = (args.output_dir or repo_root / "dist").resolve()
    return models_root, manifest, output_dir


def load_entries(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries: list[dict] = []
    entries.append(payload["edge_default"])
    for section in ("training_and_regression", "semantic_assets", "pose_assets"):
        entries.extend(payload.get(section) or [])
    seen: set[str] = set()
    for entry in entries:
        path = entry["path"]
        if path in seen:
            raise ValueError(f"duplicate manifest entry: {path}")
        seen.add(path)
    return entries


def verify_manifest_entries(entries: list[dict], models_root: Path) -> None:
    """Check files on disk against the manifest; warn on missing null entries."""
    for entry in entries:
        path = entry["path"]
        file_path = models_root / path
        if not file_path.is_file():
            if entry.get("release_asset") is None:
                print(f"WARNING: un-packaged manifest entry missing on disk: {path}")
                continue
            raise FileNotFoundError(f"packaged manifest entry missing: {file_path}")
        actual = sha256(file_path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"sha256 mismatch for {path}: manifest {entry['sha256']}, disk {actual}"
            )


def collect_edge_extras(models_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in EDGE_EXTRA_ROOTS:
        root_path = models_root / root
        if not root_path.is_dir():
            raise FileNotFoundError(f"edge extra root missing: {root_path}")
        files.extend(p for p in sorted(root_path.rglob("*")) if p.is_file())
    return files


def cross_check_fp16_readme(models_root: Path, files: list[Path]) -> None:
    readme = models_root / FP16_README
    if not readme.is_file():
        raise FileNotFoundError(f"missing {FP16_README}")
    published: dict[str, str] = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        match = DIGEST_LINE.match(line.strip())
        if match:
            published[Path(match.group(2)).as_posix()] = match.group(1)
    for file_path in files:
        relative = file_path.relative_to(models_root).as_posix()
        engine_key = relative.removeprefix("models/pose/fp16/")
        if engine_key in published:
            actual = sha256(file_path)
            if actual != published[engine_key]:
                raise ValueError(
                    f"{FP16_README} digest mismatch for {engine_key}: "
                    f"published {published[engine_key]}, disk {actual}"
                )
        # Non-engine metadata files are not documented; nothing to check.


def build_packages(entries: list[dict], models_root: Path, output_dir: Path,
                   dry_run: bool) -> dict[str, dict]:
    """Group files per release asset and return the package plan."""
    edge_extras = collect_edge_extras(models_root)
    cross_check_fp16_readme(models_root, edge_extras)

    server_files: list[Path] = []
    edge_files: list[Path] = []
    for entry in entries:
        asset = entry.get("release_asset")
        if asset == "models-server.tar.gz":
            server_files.append(models_root / entry["path"])
            for sidecar in SIDECAR_RULES.get(entry["path"], ()):
                sidecar_path = models_root / sidecar
                if not sidecar_path.is_file():
                    raise FileNotFoundError(f"sidecar missing: {sidecar_path}")
                server_files.append(sidecar_path)
        elif asset == "models-edge.tar.gz":
            edge_files.append(models_root / entry["path"])
            for sidecar in SIDECAR_RULES.get(entry["path"], ()):
                sidecar_path = models_root / sidecar
                if not sidecar_path.is_file():
                    raise FileNotFoundError(f"sidecar missing: {sidecar_path}")
                edge_files.append(sidecar_path)
    edge_files.extend(edge_extras)

    server_files = sorted(set(server_files))
    edge_files = sorted(set(edge_files))
    server_paths = {p.relative_to(models_root).as_posix() for p in server_files}
    edge_paths = {p.relative_to(models_root).as_posix() for p in edge_files}
    overlap = server_paths & edge_paths
    if overlap:
        raise ValueError(f"package path overlap: {sorted(overlap)}")

    packages = {
        "models-server.tar.gz": {"files": server_files, "manifest_name": "models/MANIFEST.server.json",
                                 "sums_name": "models/SHA256SUMS.server"},
        "models-edge.tar.gz": {"files": edge_files, "manifest_name": "models/MANIFEST.edge.json",
                               "sums_name": "models/SHA256SUMS.edge"},
    }
    _print_plan(packages, models_root, output_dir, dry_run)
    return packages


def _print_plan(packages: dict[str, dict], models_root: Path,
                output_dir: Path, dry_run: bool) -> None:
    print("Package plan:")
    for name, package in packages.items():
        total = sum(p.stat().st_size for p in package["files"])
        print(f"  {name}: {len(package['files'])} files, "
              f"{total / 1024 / 1024:.1f} MiB")
        for file_path in package["files"]:
            print(f"    - {file_path.relative_to(models_root)}")
    print(f"Output directory: {output_dir}")
    if dry_run:
        print("Dry run: nothing written.")


def build_package_metadata(name: str, files: list[Path], models_root: Path,
                           manifest: Path) -> tuple[dict, str]:
    git_commit = ""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True,
            cwd=models_root, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"

    entries = []
    for file_path in files:
        relative = file_path.relative_to(models_root).as_posix()
        entries.append({
            "path": relative,
            "sha256": sha256(file_path),
            "bytes": file_path.stat().st_size,
        })
    # Reproducible builds: honor SOURCE_DATE_EPOCH so rebuilds of the same
    # tree produce byte-identical archives.
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    created_at = (
        datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc).isoformat(timespec="seconds")
        if source_date_epoch
        else datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    metadata = {
        "schema_version": "dahua_model_package.v1",
        "package": name,
        "source_manifest": manifest.relative_to(models_root).as_posix(),
        "source_manifest_sha256": sha256(manifest),
        "git_commit": git_commit,
        "created_at": created_at,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }
    sums = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in entries
    )
    return metadata, sums


def write_archive(name: str, package: dict, models_root: Path,
                  output_dir: Path, manifest: Path) -> str:
    staging = output_dir / "staging" / name.removesuffix(".tar.gz")
    staging.mkdir(parents=True, exist_ok=True)
    for leftover in staging.iterdir():
        if leftover.is_file():
            leftover.unlink()

    metadata, sums = build_package_metadata(
        name, package["files"], models_root, manifest
    )
    staging_manifest = staging / "MANIFEST.json"
    staging_sums = staging / "SHA256SUMS"
    staging_manifest.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staging_sums.write_text(sums, encoding="utf-8")

    archive_path = output_dir / name
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=FIXED_MTIME) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for file_path in package["files"]:
                    info = tar.gettarinfo(
                        str(file_path), arcname=file_path.relative_to(models_root).as_posix()
                    )
                    info.mtime = FIXED_MTIME
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with file_path.open("rb") as stream:
                        tar.addfile(info, stream)
                for meta_path, arcname in (
                    (staging_manifest, package["manifest_name"]),
                    (staging_sums, package["sums_name"]),
                ):
                    info = tar.gettarinfo(str(meta_path), arcname=arcname)
                    info.mtime = FIXED_MTIME
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with meta_path.open("rb") as stream:
                        tar.addfile(info, stream)

    digest = sha256(archive_path)
    sidecar = output_dir / f"{name}.sha256"
    sidecar.write_text(f"{digest}  {name}\n", encoding="utf-8")
    return digest


def main() -> None:
    args = parse_args()
    models_root, manifest, output_dir = resolve_paths(args)
    entries = load_entries(manifest)
    verify_manifest_entries(entries, models_root)
    packages = build_packages(entries, models_root, output_dir, args.dry_run)
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, package in packages.items():
        digests[name] = write_archive(name, package, models_root, output_dir, manifest)
        print(f"\n{name}: sha256 {digests[name]}")
        print(f"  archive:  {output_dir / name}")
        print(f"  sidecar:  {output_dir / f'{name}.sha256'}")

    tag = "models-v1.1.0"
    print("\nPublish with:")
    print(
        "gh release create %s \\\n"
        "  %s/models-server.tar.gz %s/models-server.tar.gz.sha256 \\\n"
        "  %s/models-edge.tar.gz %s/models-edge.tar.gz.sha256 \\\n"
        "  --title \"Campus6 model assets v1.1.0 (server/edge split)\" \\\n"
        "  --notes \"See README ## Model download for contents and digests.\""
        % (tag, output_dir, output_dir, output_dir, output_dir)
    )


if __name__ == "__main__":
    main()
