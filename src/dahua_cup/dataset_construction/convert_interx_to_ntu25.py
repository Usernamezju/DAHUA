#!/usr/bin/env python3
"""Convert selected Inter-X raw skeletons to Campus6-compatible NTU25 features.

The official ``skeletons.zip`` is 23+ GB.  This program reads its ZIP central
directory through HTTP byte-range requests and downloads only P1/P2 arrays for
the sequences declared in a Campus6 mapping manifest.  It never downloads or
extracts the full archive.

Inter-X provides 64 OptiTrack joints.  The mapping retains pelvis, spine,
neck/head, shoulders, elbows, wrists and lower limbs.  The NTU hand-tip/thumb
slots use wrist coordinates because Inter-X's hand topology is not equivalent
to Kinect's five hand joints.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import time
import zlib
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests


SKELETONS_ZIP_ID = "1gj0FVC_vuK7g9rzl5yrvAaK_nqNIIglK"
ZIP_SIZE = 23_458_857_993
DEFAULT_PROXY = "http://127.0.0.1:7898"

# Inter-X OptiTrack tree from visualize/joint_viewer_tool/data_viewer.py.
# These indices retain body kinematics; NTU indices follow the project's
# MediaPipe-to-NTU25 convention.
NTU25_FROM_INTERX64 = (
    (0,),       # spine base / pelvis
    (9,),       # spine mid
    (10,),      # neck / upper spine
    (60,),      # head
    (12,),      # left shoulder
    (13,),      # left elbow
    (14,),      # left wrist
    (14,),      # left hand (wrist proxy)
    (36,),      # right shoulder
    (37,),      # right elbow
    (38,),      # right wrist
    (38,),      # right hand (wrist proxy)
    (1,),       # left hip
    (2,),       # left knee
    (3,),       # left ankle
    (4,),       # left foot
    (5,),       # right hip
    (6,),       # right knee
    (7,),       # right ankle
    (8,),       # right foot
    (10,),      # spine shoulder
    (14,),      # left hand tip (wrist proxy)
    (14,),      # left thumb (wrist proxy)
    (38,),      # right hand tip (wrist proxy)
    (38,),      # right thumb (wrist proxy)
)


class DriveRangeClient:
    """Read a public Drive file by Range, refreshing a one-use confirm URL."""

    def __init__(self, file_id: str, proxy: str, retries: int = 6):
        self.file_id = file_id
        self.proxy = proxy
        self.retries = retries

    def _session_and_confirmed_url(self):
        # gdown owns the slightly brittle Google confirmation-page parser.
        from gdown.download import _get_session, get_url_from_gdrive_confirmation

        session, _ = _get_session(
            proxy=self.proxy,
            use_cookies=True,
            user_agent=None,
            return_cookies_file=True,
        )
        initial = session.get(
            f"https://drive.google.com/uc?id={self.file_id}",
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=(20, 60),
        )
        try:
            kind = initial.headers.get("Content-Type", "")
            if "Content-Disposition" in initial.headers and not kind.startswith("text/html"):
                return session, initial.url
            html = initial.text
            if "quota exceeded" in html.lower():
                raise RuntimeError("Google Drive reports download quota exceeded")
            return session, get_url_from_gdrive_confirmation(html)
        finally:
            initial.close()

    def read(self, start: int, length: int) -> bytes:
        if start < 0 or length < 1:
            raise ValueError(f"invalid byte range: {start=}, {length=}")
        end = start + length - 1
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            session = None
            response = None
            try:
                session, url = self._session_and_confirmed_url()
                response = session.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(20, 120),
                )
                if response.status_code != 206:
                    preview = response.content[:200].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Drive did not honor bytes={start}-{end}: HTTP {response.status_code}; {preview!r}"
                    )
                payload = response.content
                if len(payload) != length:
                    raise RuntimeError(
                        f"short Drive range: wanted {length} bytes, got {len(payload)}"
                    )
                return payload
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 * attempt, 10))
            finally:
                if response is not None:
                    response.close()
                if session is not None:
                    session.close()
        raise RuntimeError(f"range request failed after {self.retries} attempts") from last_error


class RangeReader(io.RawIOBase):
    """Minimal seekable file object for ZIP central-directory parsing."""

    def __init__(self, client: DriveRangeClient, size: int):
        self.client = client
        self.size = size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if target < 0:
            raise ValueError("negative seek")
        self.position = min(target, self.size)
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        payload = self.client.read(self.position, size)
        self.position += len(payload)
        return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--zip-size", type=int, default=ZIP_SIZE)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["campus6_label"] not in {"playful_push", "conflict_push"}:
                raise ValueError(f"unexpected mapped label: {row['campus6_label']}")
            rows.append(row)
    return rows


def matching_members(archive: zipfile.ZipFile, sequence_id: str) -> dict[str, zipfile.ZipInfo]:
    suffixes = {f"/{sequence_id}/P1.npy": "P1", f"/{sequence_id}/P2.npy": "P2"}
    result = {}
    for info in archive.infolist():
        normalized = "/" + info.filename.lstrip("/")
        for suffix, person in suffixes.items():
            if normalized.endswith(suffix):
                result[person] = info
    if set(result) != {"P1", "P2"}:
        raise KeyError(f"missing P1/P2 in skeletons.zip for {sequence_id}: {sorted(result)}")
    return result


def load_zip_member(client: DriveRangeClient, info: zipfile.ZipInfo) -> np.ndarray:
    local_header = client.read(info.header_offset, 30)
    signature, _, _, method, _, _, _, _, _, name_size, extra_size = struct.unpack(
        "<4s5H3L2H", local_header
    )
    if signature != b"PK\x03\x04":
        raise RuntimeError(f"invalid local ZIP header at {info.header_offset}")
    if method != info.compress_type:
        raise RuntimeError("central/local ZIP compression type mismatch")
    data_offset = info.header_offset + 30 + name_size + extra_size
    compressed = client.read(data_offset, info.compress_size)
    if method == zipfile.ZIP_STORED:
        payload = compressed
    elif method == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"unsupported ZIP compression method: {method}")
    return np.asarray(np.load(io.BytesIO(payload), allow_pickle=False), dtype=np.float32)


def to_ntu25(skeleton: np.ndarray) -> np.ndarray:
    if skeleton.ndim != 3 or skeleton.shape[1:] != (64, 3):
        raise ValueError(f"expected Inter-X [T,64,3], got {skeleton.shape}")
    mapped = skeleton[:, NTU25_FROM_INTERX64, :].copy()
    if not np.isfinite(mapped).all():
        raise ValueError("Inter-X skeleton contains non-finite coordinates")
    return mapped


def convert_pair(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # Raw skeletons are 120 FPS.  Inter-X's official processed data uses a
    # factor-four downsample, so this keeps our feature temporal scale at 30 FPS.
    p1 = to_ntu25(p1)[::4]
    p2 = to_ntu25(p2)[::4]
    frames = min(len(p1), len(p2))
    if frames < 2:
        raise ValueError(f"sequence has too few aligned frames: {len(p1)}, {len(p2)}")
    output = np.stack([p1[:frames], p2[:frames]], axis=0)
    # Match Inter-X's official P1-origin normalization while preserving both
    # actors' positions, their relative distance and all temporal motion.
    output -= output[0, 0, 0][None, None, None, :]
    return output.astype(np.float32, copy=False)


def main():
    args = parse_args()
    rows = read_manifest(args.mapping_manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("mapping manifest is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = DriveRangeClient(SKELETONS_ZIP_ID, args.proxy)
    archive = zipfile.ZipFile(RangeReader(client, args.zip_size))
    output_rows = []
    counts = Counter()
    for index, row in enumerate(rows, start=1):
        sequence_id = row["sequence_id"]
        label = row["campus6_label"]
        target = args.output_dir / "features" / label / f"{sequence_id}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with np.load(target, allow_pickle=False) as existing:
                frames = int(existing["keypoint"].shape[1])
        else:
            members = matching_members(archive, sequence_id)
            p1 = load_zip_member(client, members["P1"])
            p2 = load_zip_member(client, members["P2"])
            keypoint = convert_pair(p1, p2)
            frames = int(keypoint.shape[1])
            np.savez_compressed(
                target,
                schema_version=np.asarray("interx_ntu25.v1"),
                keypoint=keypoint,
                keypoint_score=np.ones(keypoint.shape[:3], dtype=np.float32),
                valid_mask=np.ones(keypoint.shape[:3], dtype=bool),
                fps=np.asarray(30.0, dtype=np.float32),
                total_frames=np.asarray(frames, dtype=np.int32),
                source_dataset=np.asarray("Inter-X"),
                source_sequence_id=np.asarray(sequence_id),
                source_action=np.asarray(row["interx_action"]),
            )
        output_row = dict(row)
        output_row.update(
            {
                "feature_path": str(target),
                "skeleton_schema": "interx_ntu25.v1",
                "fps": 30.0,
                "total_frames": frames,
            }
        )
        output_rows.append(output_row)
        counts[label] += 1
        print(f"[{index}/{len(rows)}] {sequence_id} -> {target}", flush=True)

    manifest = args.output_dir / "converted_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "interx_ntu25_conversion.v1",
        "source_zip_file_id": SKELETONS_ZIP_ID,
        "samples": len(output_rows),
        "labels": dict(sorted(counts.items())),
        "output_manifest": str(manifest),
        "joint_mapping": "Inter-X OptiTrack64 -> NTU25; hand tip/thumb use wrist proxy",
    }
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
