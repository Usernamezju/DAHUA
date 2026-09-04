"""Deep Compression primitives: pruning, trained weight sharing and Huffman packing.

This module implements the three-stage procedure of Han et al., *Deep
Compression* (ICLR 2016) for PyTorch Conv2d/Linear layers.  It deliberately
keeps the sparse, codebook-based representation separate from the ordinary
PyTorch execution graph: dense CUDA kernels do not accelerate unstructured
sparsity, whereas the exported package accurately represents storage savings.
"""

from __future__ import annotations

import heapq
import io
import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class PruningReport:
    target_sparsity: float
    total: int
    kept: int
    layers: dict

    @property
    def achieved_sparsity(self) -> float:
        return 1.0 - self.kept / self.total

    def as_dict(self) -> dict:
        return {
            "target_sparsity": self.target_sparsity,
            "total": self.total,
            "kept": self.kept,
            "achieved_sparsity": self.achieved_sparsity,
            "layers": self.layers,
        }


def inference_weight_modules(model) -> Iterable[tuple[str, object]]:
    """Yield only Conv/Linear weights used by Campus6 inference.

    GAP semantic projections and the CSC auxiliary loss exist only during
    training, therefore compressing them would inflate reported deployment
    storage without affecting logits.
    """
    import torch.nn as nn

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if name.startswith("backbone.") or name == "cls_head.fc_cls":
            yield name, module


def all_conv_linear_modules(model) -> Iterable[tuple[str, object]]:
    """Yield every inference Conv2d/Linear layer of an MMDetection/MMPose model.

    RTMDet and RTMPose do not carry the Campus6-only GAP/CSC branches, so all
    of their convolution and linear layers participate in the paper's three
    Deep Compression stages.
    """
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield name, module


def _selected_modules(model, selector: Callable | None) -> list[tuple[str, object]]:
    modules = list((selector or inference_weight_modules)(model))
    if not modules:
        raise ValueError("Deep Compression selected no Conv2d/Linear weights")
    return modules


class HardMask:
    """Persistent magnitude-pruning mask with zero gradients for pruned edges."""

    def __init__(self, mask) -> None:
        import torch.nn as nn

        self.module = nn.Module()
        self.module.register_buffer("mask", mask)

        def forward(weight):
            return weight * self.module.mask

        self.module.forward = forward


def _mask_parameterization(mask):
    """Construct a tiny nn.Module without exposing PyTorch internals to callers."""
    import torch.nn as nn

    class _Mask(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("mask", value)

        def forward(self, weight):
            return weight * self.mask

    return _Mask(mask)


def apply_global_magnitude_pruning(
    model,
    target_sparsity: float,
    max_layer_sparsity: float = 0.90,
    *,
    module_selector: Callable | None = None,
) -> tuple[dict, PruningReport]:
    """Globally prune the smallest weights while preserving every GCN branch.

    A single global threshold is the paper's starting point.  Campus6 has many
    small multi-scale temporal branches, however, and their weight scales are
    not directly comparable to the large prototype blocks.  The per-layer cap
    prevents an entire branch from disappearing solely because its weights
    have smaller magnitudes; the report records the resulting exact sparsity.
    """
    import torch
    from torch.nn.utils import parametrize

    if not 0.0 < target_sparsity < 1.0 or not 0.0 < max_layer_sparsity < 1.0:
        raise ValueError("sparsities must be in (0, 1)")
    modules = _selected_modules(model, module_selector)
    flat = torch.cat([module.weight.detach().abs().flatten().cpu() for _, module in modules])
    prune_count = int(flat.numel() * target_sparsity)
    threshold = torch.kthvalue(flat, max(1, prune_count)).values
    masks, rows, kept = {}, {}, 0
    for name, module in modules:
        # Ties at the global threshold are kept.  This protects small kernels
        # from becoming all-zero while retaining magnitude-pruning semantics.
        mask = (module.weight.detach().abs() > threshold).to(module.weight.dtype)
        minimum_keep = max(1, int(round(mask.numel() * (1.0 - max_layer_sparsity))))
        if int(mask.sum().item()) < minimum_keep:
            top = module.weight.detach().abs().reshape(-1).topk(minimum_keep).indices
            mask.reshape(-1)[top] = 1
        masks[name] = mask.detach().clone()
        kept_here = int(mask.sum().item())
        kept += kept_here
        rows[name] = {
            "shape": list(module.weight.shape),
            "total": int(mask.numel()),
            "kept": kept_here,
            "sparsity": 1.0 - kept_here / mask.numel(),
        }
        parametrize.register_parametrization(module, "weight", _mask_parameterization(mask))
    report = PruningReport(target_sparsity, int(flat.numel()), kept, rows)
    return masks, report


def materialize_masks(model, *, module_selector: Callable | None = None) -> None:
    """Make masked zeros ordinary parameters before trained quantization."""
    from torch.nn.utils import parametrize

    for _name, module in _selected_modules(model, module_selector):
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


def _linear_kmeans(values, clusters: int, iterations: int = 20):
    """One-dimensional k-means with paper's linear centroid initialization."""
    import torch

    values = values.detach().float().flatten().cpu()
    if values.numel() < clusters:
        clusters = int(values.numel())
    low, high = values.min(), values.max()
    if float(high - low) == 0.0:
        return low.repeat(clusters), torch.zeros(values.numel(), dtype=torch.long)
    centers = torch.linspace(low, high, steps=clusters)
    for _ in range(iterations):
        assignments = []
        for chunk in values.split(65536):
            assignments.append((chunk[:, None] - centers[None, :]).abs().argmin(dim=1))
        assignment = torch.cat(assignments)
        counts = torch.bincount(assignment, minlength=clusters)
        sums = torch.zeros_like(centers).scatter_add_(0, assignment, values)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    assignments = []
    for chunk in values.split(65536):
        assignments.append((chunk[:, None] - centers[None, :]).abs().argmin(dim=1))
    return centers, torch.cat(assignments)


def _shared_parameterization(centroids, assignment, mask):
    import torch.nn as nn

    class _SharedWeights(nn.Module):
        def __init__(self, centers, indices, keep):
            super().__init__()
            self.centroids = nn.Parameter(centers)
            self.register_buffer("assignment", indices.long())
            self.register_buffer("mask", keep)

        def forward(self, _weight):
            return self.centroids[self.assignment].reshape_as(self.mask) * self.mask

    return _SharedWeights(centroids, assignment.reshape_as(mask), mask)


def apply_trained_weight_sharing(
    model,
    masks: dict,
    conv_bits: int,
    head_bits: int,
    kmeans_iterations: int = 20,
    *,
    module_selector: Callable | None = None,
) -> dict:
    """Attach fixed k-means assignments and trainable shared centroids."""
    import torch
    from torch.nn.utils import parametrize

    if not 2 <= conv_bits <= 8 or not 2 <= head_bits <= 8:
        raise ValueError("weight-sharing bit widths must be in [2, 8]")
    result = {}
    for name, module in _selected_modules(model, module_selector):
        mask = masks[name].to(module.weight.device)
        values = module.weight.detach()[mask.bool()]
        bits = head_bits if name == "cls_head.fc_cls" else conv_bits
        clusters = min(1 << bits, int(values.numel()))
        centers, assignment = _linear_kmeans(values, clusters, kmeans_iterations)
        full_assignment = torch.zeros(mask.numel(), dtype=torch.long, device=module.weight.device)
        full_assignment[mask.reshape(-1).bool()] = assignment.to(module.weight.device)
        parameterization = _shared_parameterization(
            centers.to(module.weight.device), full_assignment.reshape_as(mask), mask
        )
        parametrize.register_parametrization(module, "weight", parameterization)
        result[name] = {
            "bits": bits,
            "clusters": clusters,
            "nonzero": int(values.numel()),
            "parameterization": parameterization,
        }
    return result


def materialize_weight_sharing(
    model, *, module_selector: Callable | None = None
) -> None:
    from torch.nn.utils import parametrize

    for _name, module in _selected_modules(model, module_selector):
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


def _huffman_code_lengths(symbols: list[int]) -> dict[int, int]:
    counts = Counter(symbols)
    if not counts:
        return {}
    if len(counts) == 1:
        return {next(iter(counts)): 1}
    heap = [(count, index, symbol) for index, (symbol, count) in enumerate(sorted(counts.items()))]
    heapq.heapify(heap)
    serial = len(heap)
    while len(heap) > 1:
        left, right = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, (left[0] + right[0], serial, (left, right)))
        serial += 1
    lengths = {}

    def visit(node, depth):
        if isinstance(node, int):
            lengths[node] = depth
        else:
            visit(node[0][2], depth + 1)
            visit(node[1][2], depth + 1)

    visit(heap[0][2], 0)
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    code, previous, result = 0, 0, {}
    for symbol, length in sorted(lengths.items(), key=lambda row: (row[1], row[0])):
        code <<= length - previous
        result[symbol] = (code, length)
        code += 1
        previous = length
    return result


def _encode_symbols(symbols: list[int]) -> tuple[bytes, list[list[int]], int]:
    lengths = _huffman_code_lengths(symbols)
    codes = _canonical_codes(lengths)
    accumulator = bits = 0
    body = bytearray()
    for symbol in symbols:
        code, width = codes[symbol]
        accumulator = (accumulator << width) | code
        bits += width
        while bits >= 8:
            bits -= 8
            body.append((accumulator >> bits) & 0xFF)
    if bits:
        body.append((accumulator << (8 - bits)) & 0xFF)
    return bytes(body), [[symbol, length] for symbol, length in sorted(lengths.items())], bits


def export_huffman_package(
    model,
    masks: dict,
    sharing: dict,
    destination: str | Path,
    *,
    module_selector: Callable | None = None,
    format_name: str = "campus6.deep_compression.huffman.v1",
) -> dict:
    """Export a paper-style sparse weight/index Huffman package.

    The package is storage-oriented, not a claim of dense-PyTorch speedup.
    It contains a per-layer codebook plus Huffman-coded codebook assignments
    and flattened sparse-position deltas.  Non-inference GAP/CSC tensors are
    deliberately excluded.
    """
    import numpy as np
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header, body = {"format": format_name, "layers": []}, bytearray()
    raw_float_bytes = compressed_bits = nonzero_total = 0
    for name, module in _selected_modules(model, module_selector):
        parameterization = sharing[name]["parameterization"]
        mask = masks[name].detach().cpu().bool().reshape(-1)
        positions = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
        assignments = parameterization.assignment.detach().cpu().reshape(-1)[mask].tolist()
        deltas, previous = [], -1
        for position in positions:
            deltas.append(position - previous)
            previous = position
        assignment_bytes, assignment_lengths, assignment_tail = _encode_symbols(assignments)
        delta_bytes, delta_lengths, delta_tail = _encode_symbols(deltas)
        offset = len(body)
        body.extend(assignment_bytes)
        body.extend(delta_bytes)
        centroids = parameterization.centroids.detach().cpu().numpy().astype(np.float32)
        header["layers"].append({
            "name": name,
            "shape": list(module.weight.shape),
            "nonzero": len(positions),
            "centroids": centroids.tolist(),
            "assignment_huffman": assignment_lengths,
            "delta_huffman": delta_lengths,
            "assignment_bytes": len(assignment_bytes),
            "delta_bytes": len(delta_bytes),
            "assignment_tail_bits": assignment_tail,
            "delta_tail_bits": delta_tail,
            "offset": offset,
        })
        raw_float_bytes += module.weight.numel() * 4
        compressed_bits += 8 * (len(assignment_bytes) + len(delta_bytes)) + 32 * len(centroids)
        nonzero_total += len(positions)
    encoded_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    destination.write_bytes(b"DCMP1" + struct.pack("<Q", len(encoded_header)) + encoded_header + body)
    result = {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "raw_selected_weight_bytes": raw_float_bytes,
        "selected_nonzero": nonzero_total,
        "payload_estimated_bits": compressed_bits,
        "header_bytes": len(encoded_header),
        "package_vs_selected_fp32_reduction": 1.0 - destination.stat().st_size / raw_float_bytes,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def read_huffman_package(source: str | Path) -> tuple[dict, bytes]:
    """Read a ``campus6.deep_compression.huffman.v1`` package safely."""
    source = Path(source)
    raw = source.read_bytes()
    if len(raw) < 13 or raw[:5] != b"DCMP1":
        raise ValueError("not a Deep Compression Huffman package")
    header_size = struct.unpack("<Q", raw[5:13])[0]
    end = 13 + header_size
    if end > len(raw):
        raise ValueError("truncated Deep Compression Huffman header")
    try:
        header = json.loads(raw[13:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Deep Compression Huffman header") from exc
    if header.get("format") != "campus6.deep_compression.huffman.v1":
        raise ValueError(f"unsupported Deep Compression format: {header.get('format')!r}")
    if not isinstance(header.get("layers"), list) or not header["layers"]:
        raise ValueError("Deep Compression package has no layers")
    return header, raw[end:]


def _decode_symbols(encoded: bytes, lengths: list[list[int]], count: int) -> list[int]:
    """Decode the canonical, MSB-first Huffman stream emitted above."""
    if count < 0:
        raise ValueError("negative Huffman symbol count")
    if count == 0:
        return []
    codes = _canonical_codes({int(symbol): int(width) for symbol, width in lengths})
    reverse = {(code, width): symbol for symbol, (code, width) in codes.items()}
    if not reverse:
        raise ValueError("Huffman stream has symbols but no code table")
    maximum = max(width for _code, width in reverse)
    result, code, width = [], 0, 0
    for byte in encoded:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            width += 1
            symbol = reverse.get((code, width))
            if symbol is not None:
                result.append(symbol)
                if len(result) == count:
                    return result
                code = width = 0
            elif width > maximum:
                raise ValueError("invalid Huffman code in Deep Compression package")
    raise ValueError("truncated Huffman stream in Deep Compression package")


def huffman_weight_keys(source: str | Path) -> set[str]:
    """Return state-dict keys materialized from the archive at load time."""
    header, _body = read_huffman_package(source)
    return {f"{row['name']}.weight" for row in header["layers"]}


def campus6_training_only_state_keys(model) -> set[str]:
    """Return GAP/CSC state that is never read by Campus6 evaluation.

    ``RecognizerGCNGAP.forward_train`` alone uses ``semantic_projection`` and
    ``semantic_text``; ``cls_head.csc_loss`` is likewise a training loss.  The
    inherited inference path only uses the backbone and classifier logits, so
    carrying those tensors in a deployment sidecar wastes storage and makes
    no numerical contribution to a prediction.
    """
    state = model.state_dict()
    prefixes = ("semantic_projection.", "cls_head.csc_loss.")
    return {
        name
        for name in state
        if name == "semantic_text" or name.startswith(prefixes)
    }


def export_huffman_runtime_residual(
    model,
    package: str | Path,
    destination: str | Path,
    *,
    excluded_state_keys: Iterable[str] = (),
) -> dict:
    """Persist only state that cannot be reconstructed from a Huffman package.

    The package reconstructs every compressed Conv/Linear weight.  BatchNorm
    parameters and buffers, biases, and the small non-compressed inference
    tensors remain in this sidecar.  Callers may exclude explicitly declared
    training-only state; its names are retained in metadata so a loader can
    reject a sidecar that omits an inference tensor.  The combination is a
    self-contained deployment artifact without a dense copy of compressed
    weights or unused training branches.
    """
    import torch

    selected = huffman_weight_keys(package)
    state = model.state_dict()
    missing = selected.difference(state)
    if missing:
        raise ValueError(f"Huffman layers absent from model state: {sorted(missing)[:3]}")
    excluded = {str(name) for name in excluded_state_keys}
    unknown = excluded.difference(state)
    if unknown:
        raise ValueError(f"residual exclusions absent from model state: {sorted(unknown)[:3]}")
    overlap = excluded.intersection(selected)
    if overlap:
        raise ValueError(f"residual exclusions overlap Huffman weights: {sorted(overlap)[:3]}")
    residual = {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name not in selected and name not in excluded
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": residual,
            "meta": {
                "format": "campus6.deep_compression.residual.v1",
                "huffman_package": Path(package).name,
                "huffman_weight_keys": len(selected),
                "excluded_training_only_keys": sorted(excluded),
            },
        },
        destination,
    )
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "state_keys": len(residual),
        "excluded_huffman_weight_keys": len(selected),
        "excluded_training_only_keys": len(excluded),
    }


def load_huffman_weights(model, source: str | Path) -> dict:
    """Materialize Huffman/codebook sparse weights into an initialized model.

    This is deliberately a load-time reconstruction.  Standard dense PyTorch
    kernels then execute the exact compressed model; a sparse CUDA kernel is
    a separate optimization and is not implied by the storage compression.
    """
    import torch

    header, body = read_huffman_package(source)
    modules = dict(model.named_modules())
    restored = 0
    for row in header["layers"]:
        name = str(row["name"])
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"Huffman layer is missing from model: {name}")
        shape = tuple(int(value) for value in row["shape"])
        if tuple(module.weight.shape) != shape:
            raise ValueError(f"Huffman layer shape mismatch for {name}: {tuple(module.weight.shape)} != {shape}")
        nonzero, offset = int(row["nonzero"]), int(row["offset"])
        assignment_size, delta_size = int(row["assignment_bytes"]), int(row["delta_bytes"])
        end = offset + assignment_size + delta_size
        if offset < 0 or end > len(body):
            raise ValueError(f"Huffman payload range is invalid for {name}")
        assignment = _decode_symbols(body[offset : offset + assignment_size], row["assignment_huffman"], nonzero)
        deltas = _decode_symbols(body[offset + assignment_size : end], row["delta_huffman"], nonzero)
        centroids = torch.as_tensor(row["centroids"], dtype=module.weight.dtype, device=module.weight.device)
        indices = torch.as_tensor(assignment, dtype=torch.long, device=module.weight.device)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= centroids.numel()):
            raise ValueError(f"Huffman codebook index out of range for {name}")
        positions, previous = [], -1
        for delta in deltas:
            if delta <= 0:
                raise ValueError(f"Huffman sparse delta is invalid for {name}")
            previous += int(delta)
            positions.append(previous)
        if positions and positions[-1] >= module.weight.numel():
            raise ValueError(f"Huffman sparse position out of range for {name}")
        target = torch.zeros(module.weight.numel(), dtype=module.weight.dtype, device=module.weight.device)
        if positions:
            target[torch.as_tensor(positions, dtype=torch.long, device=module.weight.device)] = centroids[indices]
        with torch.no_grad():
            module.weight.copy_(target.reshape(shape))
        restored += 1
    return {"format": header["format"], "layers_restored": restored, "package": str(source)}
