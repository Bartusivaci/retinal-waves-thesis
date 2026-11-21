"""Map a cartesian movie to a full-network FlyVis stimulus via BoxEye photoreceptors.

Moved here from the former scripts/train_photoreceptor_finalstate.py so the edge trainer
does not depend on old experiment scripts. Behaviour is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from flyvis.datasets.rendering import BoxEye


class PhotoreceptorWaveMapper:
    """BoxEye renders frames to 721 hexals; external input reaches only photoreceptors.

    Call with frames [1,T,1,H,W] (or [T,1,H,W]) in 0..1; returns stimulus [1,T,n_nodes].
    """

    def __init__(self, network, gain: float = 1.0):
        self.network = network
        self.eye = BoxEye(extent=15, kernel_size=13)
        nodes = network.connectome.nodes.to_df().sort_values('index')
        roles = nodes['role'].astype(str).to_numpy()
        self.photo_indices = torch.as_tensor(np.flatnonzero(roles == 'input'), dtype=torch.long)
        if not len(self.photo_indices):
            raise RuntimeError('No input-role photoreceptors found')
        self.n_nodes = int(network.n_nodes)
        self.gain = float(gain)

    @torch.no_grad()
    def __call__(self, frames):
        if frames.ndim == 4:  # [T,1,H,W]
            frames = frames.unsqueeze(0)
        if frames.ndim != 5 or frames.shape[2] != 1 or frames.shape[0] != 1:
            raise ValueError(f'Expected [1,T,1,H,W], got {tuple(frames.shape)}')
        eye_device = self.eye.conv.weight.device
        rendered = self.eye(frames[:, :, 0].float().to(eye_device)).detach().cpu()
        if rendered.shape[-1] != 721:
            raise RuntimeError(f'Unexpected BoxEye output: {tuple(rendered.shape)}')
        self.network.stimulus.zero(n_samples=1, n_frames=frames.shape[1])
        self.network.stimulus.add_input(rendered * self.gain)
        return self.network.stimulus().detach()

    @torch.no_grad()
    def filtered(self, frames):
        """Box-filtered cartesian frames [T, 391, 391] (what the photoreceptors sample).
        Hexal i reads pixel (receptor_centers[i] + 195); see pixel_centers()."""
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        eye_device = self.eye.conv.weight.device
        out = self.eye(frames[:, :, 0].float().to(eye_device), hex_sample=False)
        return out[0].detach()

    def pixel_centers(self, size):
        """(y, x) pixel coordinates [721, 2] of each hexal in a filtered frame of the given size.
        Column i here = BoxEye hexal i = decoder output column i (verified empirically)."""
        return (self.eye.receptor_centers + torch.tensor([size // 2, size // 2],
                                                         device=self.eye.receptor_centers.device)).float()
