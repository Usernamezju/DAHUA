#!/usr/bin/env python3
"""Build NTU25 features from Inter-X processed SMPL-X motion parameters."""
import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,12,12,12,13,14,16,17,18,19])
OFFSETS = np.array([
    (0,0,0),(.09,-.10,0),(-.09,-.10,0),(0,.12,0),(0,-.42,0),(0,-.42,0),
    (0,.12,0),(0,-.42,0),(0,-.42,0),(0,.12,0),(0,-.06,.12),(0,-.06,.12),
    (0,.12,0),(.12,.04,0),(-.12,.04,0),(0,.16,0),(.15,0,0),(-.15,0,0),
    (.28,0,0),(-.28,0,0),(.25,0,0),(-.25,0,0)], dtype=np.float32)
NTU = np.array([0,6,12,15,16,18,20,20,17,19,21,21,1,4,7,10,2,5,8,11,9,20,20,21,21])


def rotations(v):
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    axis = np.divide(v, theta, out=np.zeros_like(v), where=theta > 1e-8)
    x, y, z = axis.T
    k = np.zeros((len(v), 3, 3), dtype=np.float32)
    k[:,0,1], k[:,0,2], k[:,1,0], k[:,1,2], k[:,2,0], k[:,2,1] = -z,y,z,-x,-y,x
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), k.shape)
    a = theta[:,None]
    return eye + np.sin(a)*k + (1-np.cos(a))*(k @ k)


def person_to_body(person):
    if person.ndim != 3 or person.shape[1:] != (56,3) or not np.isfinite(person).all():
        raise ValueError(f"bad Inter-X person array: {person.shape}")
    t = len(person)
    local = np.stack([rotations(person[:,j]) for j in range(22)], axis=1)
    world = np.empty_like(local)
    joints = np.empty((t,22,3), dtype=np.float32)
    world[:,0], joints[:,0] = local[:,0], person[:,55]
    for j in range(1,22):
        p = PARENTS[j]
        world[:,j] = world[:,p] @ local[:,j]
        joints[:,j] = joints[:,p] + np.einsum("tij,j->ti", world[:,p], OFFSETS[j])
    return joints


def convert(data):
    if data.ndim != 3 or data.shape[1:] != (56,6):
        raise ValueError(f"bad Inter-X motion array: {data.shape}")
    keypoint = np.stack([person_to_body(data[:,:,:3])[:,NTU], person_to_body(data[:,:,3:])[:,NTU]])
    keypoint -= keypoint[0,0,0][None,None,None,:]
    return keypoint.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping-manifest", type=Path, required=True)
    p.add_argument("--interx-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.mapping_manifest.read_text().splitlines() if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files, results, counts = {}, [], Counter()
    try:
        for n, row in enumerate(rows, 1):
            h5path = (args.interx_root / row["source_h5"]).resolve()
            h5 = files.setdefault(str(h5path), h5py.File(h5path, "r"))
            sid, label = row["sequence_id"], row["campus6_label"]
            dest = args.output_dir/"features"/label/f"{sid}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                with np.load(dest, allow_pickle=False) as z: frames = int(z["keypoint"].shape[1])
            else:
                keypoint = convert(np.asarray(h5[sid], dtype=np.float32))
                frames = int(keypoint.shape[1])
                np.savez_compressed(dest, schema_version=np.asarray("interx_smplx_fk_ntu25.v1"),
                    keypoint=keypoint, keypoint_score=np.ones(keypoint.shape[:3],np.float32),
                    valid_mask=np.ones(keypoint.shape[:3],bool), fps=np.asarray(30,np.float32),
                    total_frames=np.asarray(frames,np.int32), source_dataset=np.asarray("Inter-X"),
                    source_sequence_id=np.asarray(sid), source_action=np.asarray(row["interx_action"]))
            record = dict(row, feature_path=str(dest), skeleton_schema="interx_smplx_fk_ntu25.v1",
                total_frames=frames, conversion_note="SMPL-X body FK proxy; hand tip/thumb use wrist")
            results.append(record); counts[label] += 1
            print(f"[{n}/{len(rows)}] {sid}", flush=True)
    finally:
        for f in files.values(): f.close()
    manifest = args.output_dir/"converted_manifest.jsonl"
    with manifest.open("w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+chr(10))
    report = {"schema_version":"interx_smplx_fk_ntu25.v1","samples":len(results),
        "labels":dict(sorted(counts.items())),"source":"Inter-X processed test.h5 + val.h5",
        "representation":"SMPL-X body forward-kinematics proxy",
        "limitations":"No raw OptiTrack data; hand tip/thumb use wrist proxies.","output_manifest":str(manifest)}
    (args.output_dir/"conversion_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10))
    print(json.dumps(report,ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
