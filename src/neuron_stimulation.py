"""Map retinal-wave video frames to a selected FlyVis neuronal population.

Input: frames [B,T,1,H,W] or [T,1,H,W], values nominally in [0,1].
Output: full neural external-drive buffer [B,T,N] with ONLY targets populated.

The values are imposed EXTERNAL DRIVE, not clamped neuronal activity.
Coordinate conventions and gain must be validated experimentally.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


class NeuronWaveMapper:
    def __init__(self, network, target_types: Sequence[str] = ("L1",), gain: float = 1.0):
        if not target_types:
            raise ValueError("Provide at least one target cell type")
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("gain must be a finite, nonnegative number")
        self.network = network
        self.target_types = tuple(target_types)
        self.gain = float(gain)

        nodes = network.connectome.nodes.to_df().sort_values("index")
        # Use explicit indices; dataframe row order is not guaranteed by every backend.
        ids = nodes["index"].to_numpy(dtype=np.int64)
        n_nodes = int(network.n_nodes)
        if len(ids) != n_nodes or not np.array_equal(ids, np.arange(n_nodes)):
            raise ValueError("Connectome node indices are not a contiguous [0,N) range")

        kinds = nodes["type"].astype(str).to_numpy()
        roles = nodes["role"].astype(str).to_numpy()
        target_mask = np.isin(kinds, self.target_types)
        target = np.flatnonzero(target_mask).astype(np.int64)
        if not len(target):
            raise ValueError(f"No target neurons of type {self.target_types} found")
        # Covers all cells marked 'input' by connectome, not only R1-R6.
        photoreceptors = np.flatnonzero(roles == "input").astype(np.int64)
        if not len(photoreceptors):
            raise ValueError("Could not identify input neurons by connectome role")
        if np.intersect1d(target, photoreceptors).size:
            raise ValueError("Targets overlap with photoreceptor/input neurons")
        self.target_indices = torch.tensor(target, dtype=torch.long)
        self.photoreceptor_indices = torch.tensor(photoreceptors, dtype=torch.long)
        self.n_nodes = n_nodes

        # FlyVis hex axial coordinates (u,v). Turn these into Cartesian sampling
        # coordinates, using one common *connectome-wide* spatial extent.
        u = nodes["u"].to_numpy(dtype=np.float64)
        v = nodes["v"].to_numpy(dtype=np.float64)
        x = u + 0.5 * v
        y = (math.sqrt(3) / 2.0) * v
        x_span, y_span = np.ptp(x), np.ptp(y)
        if x_span <= 0 or y_span <= 0:
            raise ValueError("Invalid connectome spatial geometry")
        # Match grid_sample coordinates [-1,1]. Orientation is a convention,
        # not a proven physical alignment between mammal retina and fly eye.
        grid_x = 2.0 * (x[target] - x.min()) / x_span - 1.0
        grid_y = 2.0 * (y[target] - y.min()) / y_span - 1.0
        self.grid = torch.tensor(
            np.column_stack((grid_x, grid_y)), dtype=torch.float32
        ).reshape(1, 1, len(target), 2)

    @property
    def n_targets(self):
        return self.target_indices.numel()

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Return full-neuron external drive [B,T,N] (no weights changed)."""
        if frames.ndim == 4:  # [T,1,H,W]
            frames = frames.unsqueeze(0)
        if frames.ndim != 5 or frames.shape[2] != 1:
            raise ValueError(f"Expected [B,T,1,H,W], got {tuple(frames.shape)}")
        if not torch.is_floating_point(frames):
            frames = frames.float()
        if not torch.isfinite(frames).all():
            raise ValueError("Frames contain NaN/Inf")
        b, t, _, h, w = frames.shape
        if h < 2 or w < 2:
            raise ValueError("Frame must be at least 2x2")
        # grid_sample treats B*T as a batch of images. Bilinear interpolation
        # produces one drive value per target neuron per time step.
        images = frames.reshape(b * t, 1, h, w)
        grid = self.grid.to(device=frames.device, dtype=frames.dtype).expand(b * t, -1, -1, -1)
        values = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)
        values = values[:, 0, 0, :].reshape(b, t, self.n_targets)
        stimulus = frames.new_zeros((b, t, self.n_nodes))
        targets = self.target_indices.to(frames.device)
        stimulus[:, :, targets] = values * self.gain
        return stimulus

    def verify(self, stimulus: torch.Tensor) -> dict:
        if stimulus.ndim != 3 or stimulus.shape[-1] != self.n_nodes:
            raise ValueError("Incorrect full-network stimulus shape")
        photo = self.photoreceptor_indices.to(stimulus.device)
        targets = self.target_indices.to(stimulus.device)
        other_mask = torch.ones(self.n_nodes, dtype=torch.bool, device=stimulus.device)
        other_mask[targets] = False
        if not torch.all(stimulus[:, :, photo] == 0):
            raise AssertionError("Photoreceptor external drive is nonzero")
        if not torch.all(stimulus[:, :, other_mask] == 0):
            raise AssertionError("Unexpected non-target stimulation")
        return {
            "targets": self.n_targets,
            "photoreceptors": int(photo.numel()),
            "nonzero_target_values": int(torch.count_nonzero(stimulus[:, :, targets]).item()),
            "maximum_external_drive": float(stimulus.abs().max().item()),
        }
