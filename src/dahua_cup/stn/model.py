"""Set-aware, rotation-free spatial-temporal transformer network.

The network is a compact YOLO localization student.  It internally predicts a
set of at most two people, merges the active person boxes into one group crop,
and applies a constrained spatial transform.  MediaPipe and ProtoGCN are not
part of this model or its training graph.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized center boxes to corner boxes."""
    if boxes.device.type == "cpu" and boxes.dtype == torch.float16:
        return cxcywh_to_xyxy(boxes.float()).to(dtype=boxes.dtype)
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized corner boxes to center boxes."""
    if boxes.device.type == "cpu" and boxes.dtype == torch.float16:
        return xyxy_to_cxcywh(boxes.float()).to(dtype=boxes.dtype)
    left, top, right, bottom = boxes.unbind(dim=-1)
    return torch.stack(
        (
            (left + right) / 2,
            (top + bottom) / 2,
            (right - left).clamp_min(0),
            (bottom - top).clamp_min(0),
        ),
        dim=-1,
    )


def square_group_boxes(
    boxes: torch.Tensor,
    *,
    context_factor: float = 1.25,
    minimum_fraction: float = 0.25,
) -> torch.Tensor:
    """Expand normalized group boxes to in-frame square crops."""
    if boxes.shape[-1] != 4:
        raise ValueError("group boxes must end with four xyxy coordinates")
    if boxes.device.type == "cpu" and boxes.dtype == torch.float16:
        return square_group_boxes(
            boxes.float(),
            context_factor=context_factor,
            minimum_fraction=minimum_fraction,
        ).to(dtype=boxes.dtype)
    if context_factor < 1.0:
        raise ValueError("context_factor must be at least one")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be in (0, 1]")
    center = xyxy_to_cxcywh(boxes)
    side = (
        torch.maximum(center[..., 2], center[..., 3]) * context_factor
    ).clamp(min=minimum_fraction, max=1.0)
    half = side / 2
    center_x = center[..., 0].clamp(min=half, max=1.0 - half)
    center_y = center[..., 1].clamp(min=half, max=1.0 - half)
    return torch.stack(
        (
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def group_boxes_from_queries(
    boxes: torch.Tensor,
    presence_logits: torch.Tensor,
    *,
    context_factor: float = 1.25,
    minimum_fraction: float = 0.25,
    temperature: float = 0.05,
    hard: bool = False,
    presence_threshold: float = 0.5,
) -> torch.Tensor:
    """Merge ``[B,T,Q,4]`` person boxes into one ``[B,T,4]`` crop."""
    if boxes.ndim != 4 or boxes.shape[-1] != 4:
        raise ValueError("person boxes must have [B,T,Q,4] shape")
    if boxes.device.type == "cpu" and boxes.dtype == torch.float16:
        return group_boxes_from_queries(
            boxes.float(),
            presence_logits.float(),
            context_factor=context_factor,
            minimum_fraction=minimum_fraction,
            temperature=temperature,
            hard=hard,
            presence_threshold=presence_threshold,
        ).to(dtype=boxes.dtype)
    if presence_logits.shape != boxes.shape[:-1]:
        raise ValueError("presence logits must match person box queries")
    if boxes.shape[2] < 1:
        raise ValueError("at least one person query is required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    probabilities = torch.sigmoid(presence_logits)
    if hard:
        active = probabilities >= presence_threshold
        left_values = boxes[..., 0]
        top_values = boxes[..., 1]
        right_values = boxes[..., 2]
        bottom_values = boxes[..., 3]
        left = torch.where(
            active, left_values, torch.ones_like(left_values)
        ).amin(dim=2)
        top = torch.where(
            active, top_values, torch.ones_like(top_values)
        ).amin(dim=2)
        right = torch.where(
            active, right_values, torch.zeros_like(right_values)
        ).amax(dim=2)
        bottom = torch.where(
            active, bottom_values, torch.zeros_like(bottom_values)
        ).amax(dim=2)
        any_active = active.any(dim=2)
    else:
        weights = probabilities.clamp_min(1e-6)
        log_weights = torch.log(weights) - torch.log(
            weights.sum(dim=2, keepdim=True)
        )

        def soft_edge(values: torch.Tensor, sign: float) -> torch.Tensor:
            scaled = log_weights + sign * values / temperature
            result = torch.logsumexp(scaled, dim=2) * temperature
            return result if sign > 0 else -result

        left = soft_edge(boxes[..., 0], -1.0)
        top = soft_edge(boxes[..., 1], -1.0)
        right = soft_edge(boxes[..., 2], 1.0)
        bottom = soft_edge(boxes[..., 3], 1.0)
        any_active = 1.0 - torch.prod(1.0 - probabilities, dim=2)

    merged = torch.stack((left, top, right, bottom), dim=-1)
    full_frame = torch.zeros_like(merged)
    full_frame[..., 2:] = 1.0
    if hard:
        merged = torch.where(any_active[..., None], merged, full_frame)
    else:
        merged = (
            any_active[..., None] * merged
            + (1.0 - any_active[..., None]) * full_frame
        )
    return square_group_boxes(
        merged,
        context_factor=context_factor,
        minimum_fraction=minimum_fraction,
    )


def group_boxes_to_parameters(boxes: torch.Tensor) -> torch.Tensor:
    """Convert square xyxy crops to ``fraction, tx, ty`` parameters."""
    center = xyxy_to_cxcywh(boxes)
    fraction = torch.maximum(center[..., 2], center[..., 3])
    tx = 2.0 * center[..., 0] - 1.0
    ty = 2.0 * center[..., 1] - 1.0
    return torch.stack((fraction, tx, ty), dim=-1)


class _Mobile3DBlock(nn.Module):
    """Depthwise-separable 3D block with spatial-only downsampling."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        spatial_stride: int,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=(1, spatial_stride, spatial_stride),
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm3d(input_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SetAwareGroupSTN(nn.Module):
    """Locate zero, one, or two people and return one group crop.

    Inputs use ``[B,C,T,H,W]``.  Individual query boxes are an internal
    distillation target only; the public spatial-transform result always uses
    their single union crop.
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        feature_channels: int = 48,
        maximum_people: int = 2,
        minimum_box_fraction: float = 0.01,
        minimum_crop_fraction: float = 0.25,
        context_factor: float = 1.25,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if maximum_people != 2:
            raise ValueError("the exact set matcher currently supports two queries")
        if feature_channels % 4:
            raise ValueError("feature_channels must be divisible by four")
        if not 0.0 < minimum_box_fraction < 1.0:
            raise ValueError("minimum_box_fraction must be in (0, 1)")
        self.maximum_people = maximum_people
        self.minimum_box_fraction = float(minimum_box_fraction)
        self.minimum_crop_fraction = float(minimum_crop_fraction)
        self.context_factor = float(context_factor)

        # Two fixed coordinate channels retain absolute location information.
        self.stem = nn.Sequential(
            nn.Conv3d(
                input_channels + 2,
                16,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),
        )
        self.stage2 = _Mobile3DBlock(16, 32, spatial_stride=2)
        self.stage3 = _Mobile3DBlock(32, 64, spatial_stride=2)
        self.stage4 = _Mobile3DBlock(64, 96, spatial_stride=2)

        self.lateral2 = nn.Conv3d(32, feature_channels, kernel_size=1)
        self.lateral3 = nn.Conv3d(64, feature_channels, kernel_size=1)
        self.lateral4 = nn.Conv3d(96, feature_channels, kernel_size=1)
        self.fpn_smooth = _Mobile3DBlock(
            feature_channels,
            feature_channels,
            spatial_stride=1,
        )
        self.occupancy_head = nn.Conv3d(
            feature_channels, 1, kernel_size=1
        )
        self.position_projection = nn.Linear(2, feature_channels, bias=False)
        self.person_queries = nn.Parameter(
            torch.empty(maximum_people, feature_channels)
        )
        self.query_decoder = nn.TransformerDecoderLayer(
            d_model=feature_channels,
            nhead=4,
            dim_feedforward=feature_channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_refinement = nn.Sequential(
            nn.Conv1d(
                feature_channels,
                feature_channels,
                kernel_size=5,
                padding=2,
                groups=feature_channels,
                bias=False,
            ),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                feature_channels,
                feature_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(inplace=True),
        )
        self.presence_head = nn.Linear(feature_channels, 1)
        self.box_head = nn.Sequential(
            nn.Linear(feature_channels, feature_channels),
            nn.SiLU(inplace=True),
            nn.Linear(feature_channels, 4),
        )
        self._initialize_predictions()

    def _initialize_predictions(self) -> None:
        nn.init.normal_(self.person_queries, std=0.02)
        nn.init.constant_(self.presence_head.bias, -1.0)
        last = self.box_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @staticmethod
    def _coordinate_channels(video: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = video.shape
        y_axis = torch.linspace(
            -1.0, 1.0, height, device=video.device, dtype=video.dtype
        )
        x_axis = torch.linspace(
            -1.0, 1.0, width, device=video.device, dtype=video.dtype
        )
        y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
        coordinates = torch.stack((x_grid, y_grid), dim=0)
        return coordinates[None, :, None].expand(
            batch, 2, frames, height, width
        )

    def _feature_pyramid(self, video: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((video, self._coordinate_channels(video)), dim=1)
        stage1 = self.stem(inputs)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        target_size = stage2.shape[2:]
        pyramid = self.lateral2(stage2)
        pyramid = pyramid + F.interpolate(
            self.lateral3(stage3),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
        pyramid = pyramid + F.interpolate(
            self.lateral4(stage4),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
        return self.fpn_smooth(pyramid)

    def _decode_queries(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = features.shape
        memory = (
            features.permute(0, 2, 3, 4, 1)
            .reshape(batch * frames, height * width, channels)
        )
        y_axis = torch.linspace(
            -1.0, 1.0, height, device=features.device, dtype=features.dtype
        )
        x_axis = torch.linspace(
            -1.0, 1.0, width, device=features.device, dtype=features.dtype
        )
        y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
        positions = torch.stack((x_grid, y_grid), dim=-1).reshape(-1, 2)
        memory = memory + self.position_projection(positions)[None]
        queries = self.person_queries[None].expand(
            batch * frames, -1, -1
        )
        decoded = self.query_decoder(queries, memory)
        decoded = decoded.reshape(
            batch, frames, self.maximum_people, channels
        )
        temporal = (
            decoded.permute(0, 2, 3, 1)
            .reshape(batch * self.maximum_people, channels, frames)
        )
        temporal = temporal + self.temporal_refinement(temporal)
        return (
            temporal.reshape(
                batch, self.maximum_people, channels, frames
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def predict(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(
                "STN input must have [B,C,T,H,W] shape, "
                f"got {tuple(video.shape)}"
            )
        features = self._feature_pyramid(video)
        query_features = self._decode_queries(features)
        presence_logits = self.presence_head(query_features).squeeze(-1)
        raw_boxes = torch.sigmoid(self.box_head(query_features))
        box_sizes = self.minimum_box_fraction + (
            1.0 - self.minimum_box_fraction
        ) * raw_boxes[..., 2:]
        boxes_cxcywh = torch.cat(
            (raw_boxes[..., :2], box_sizes), dim=-1
        )
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
        group_boxes = group_boxes_from_queries(
            boxes_xyxy,
            presence_logits,
            context_factor=self.context_factor,
            minimum_fraction=self.minimum_crop_fraction,
            hard=not self.training,
        )
        parameters = group_boxes_to_parameters(group_boxes)
        theta = self.parameters_to_theta(parameters)
        return {
            "presence_logits": presence_logits,
            "boxes_cxcywh": boxes_cxcywh,
            "boxes_xyxy": boxes_xyxy,
            "group_boxes_xyxy": group_boxes,
            "parameters": parameters,
            "theta": theta,
            "occupancy_logits": self.occupancy_head(features),
        }

    @staticmethod
    def parameters_to_theta(parameters: torch.Tensor) -> torch.Tensor:
        """Build matrices that cannot express rotation or shear."""
        if parameters.ndim != 3 or parameters.shape[-1] != 3:
            raise ValueError("STN parameters must have [B,T,3] shape")
        fraction, tx, ty = parameters.unbind(dim=-1)
        theta = parameters.new_zeros((*parameters.shape[:2], 2, 3))
        theta[..., 0, 0] = fraction
        theta[..., 1, 1] = fraction
        theta[..., 0, 2] = tx
        theta[..., 1, 2] = ty
        return theta

    @staticmethod
    def sample(
        video: torch.Tensor,
        theta: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Apply per-frame group crops to ``[B,C,T,H,W]`` video."""
        if video.ndim != 5:
            raise ValueError("video must have [B,C,T,H,W] shape")
        batch, channels, frames, height, width = video.shape
        if theta.shape != (batch, frames, 2, 3):
            raise ValueError("theta must match [B,T,2,3]")
        output_height, output_width = output_size or (height, width)
        frame_batch = (
            video.permute(0, 2, 1, 3, 4)
            .reshape(batch * frames, channels, height, width)
        )
        flat_theta = theta.reshape(batch * frames, 2, 3)
        grid = F.affine_grid(
            flat_theta,
            size=(
                batch * frames,
                channels,
                output_height,
                output_width,
            ),
            align_corners=False,
        )
        sampled = F.grid_sample(
            frame_batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return (
            sampled.reshape(
                batch,
                frames,
                channels,
                output_height,
                output_width,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def forward(
        self,
        video: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
        apply_transform: bool = True,
    ) -> dict[str, torch.Tensor]:
        outputs = self.predict(video)
        if apply_transform:
            outputs["video"] = self.sample(
                video, outputs["theta"], output_size=output_size
            )
        return outputs


# Compatibility name for code written against the first STN prototype.
SpatialTemporalCropSTN = SetAwareGroupSTN


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
