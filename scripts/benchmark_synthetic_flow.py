"""Synthetic optical-flow sanity benchmark (no FlyVis, no dataset required).

Uses EXACT warp_border, photo_loss, flow_smoothness from
scripts.train_gap5_10000, so any change there affects this benchmark too.

From repository root:
  python -m scripts.benchmark_synthetic_flow --device cuda:0
  python -m scripts.benchmark_synthetic_flow --device cpu --cnn-steps 40

Stage A: known-flow/oracle warping and brightness-constancy checks.
Stage B: optimize a low-dimensional flow model using the *same photometric loss*.
Stage C: overfit a tiny pair-conditioned CNN on a SINGLE synthetic pair.

Neither stage B nor C is supervised with ground-truth flow; ground truth is only
used to generate targets and evaluate the resulting estimates. The CNN is a
sanity test, NOT the same decoder as PairFlowHead and NOT a generalization test.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_gap5_10000 import warp_border, photo_loss, flow_smoothness

ROOT = Path(__file__).resolve().parent.parent


def make_source(side, device):
    """Generate a textured synthetic image on the selected device."""

    device = torch.device(device)

    # Generator and tensors must use the same device.
    g = torch.Generator(device=device).manual_seed(2026)

    yy, xx = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij"
    )

    x = xx.float()
    y = yy.float()

    blobs = torch.zeros(side, side, device=device)

    for cx, cy, sigma, amp in [
        (18, 18, 8, 0.8),
        (42, 21, 5, 1.0),
        (34, 43, 9, 0.7),
        (15, 46, 4, 0.65),
        (48, 49, 6, 0.9),
        (28, 31, 3, 1.1),
    ]:
        blobs += amp * torch.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )

    noise = torch.rand(
        (1, 1, side, side),
        generator=g,
        device=device
    )

    smooth_noise = F.avg_pool2d(
        noise,
        kernel_size=5,
        stride=1,
        padding=2
    )[0, 0]

    image = (
        0.12
        + 0.63 * (blobs / blobs.max())
        + 0.25 * smooth_noise
    )

    return image.clamp(0, 1)[None, None]


def gt_translation(side, device, delta=3.0):
    # Image content moves right by delta; backward flow samples delta pixels left.
    flow = torch.zeros(1, 2, side, side, device=device)
    flow[:, 0] = -delta
    return flow


def gt_expansion(side, device, scale=1.08):
    # Target image is magnified around center; target->source mapping is inverse scale.
    yy, xx = torch.meshgrid(torch.arange(side, device=device),
                             torch.arange(side, device=device), indexing='ij')
    c = (side - 1) / 2
    flow = torch.zeros(1, 2, side, side, device=device)
    flow[:, 0] = (1/scale - 1) * (xx - c)
    flow[:, 1] = (1/scale - 1) * (yy - c)
    return flow


def score(source, target, flow, gt, activity_threshold=.23):
    with torch.no_grad():
        l = float(photo_loss(source, target, flow).item())
        zero = float(photo_loss(source, target, torch.zeros_like(flow)).item())
        # Evaluate EPE in active regions, otherwise background dominates metrics.
        mask = (torch.maximum(source, target) > activity_threshold)[0, 0]
        epe = (flow - gt).square().sum(dim=1).sqrt()[0]
        active_epe = float(epe[mask].mean().item()) if mask.any() else float(epe.mean().item())
        return dict(photo_loss=l, zero_photo_loss=zero,
                    improvement_over_zero_pct=100*(zero-l)/max(zero, 1e-12),
                    active_epe_px=active_epe, mean_abs_flow_px=float(flow.abs().mean().item()))


def fit_parametric(source, target, gt, kind, steps, lr):
    device = source.device
    side = source.shape[-1]
    coefficient = nn.Parameter(torch.zeros(2 if kind == 'translation' else 1, device=device))
    optim = torch.optim.Adam([coefficient], lr=lr)
    history = []
    for k in range(steps):
        if kind == 'translation':
            flow = coefficient.view(1, 2, 1, 1).expand_as(gt)
        else:
            # Learn inverse-scale factor only; real radial direction is fixed.
            yy, xx = torch.meshgrid(torch.arange(side, device=device),
                                     torch.arange(side, device=device), indexing='ij')
            c = (side-1)/2
            template = torch.stack((xx-c, yy-c)).float()[None]
            flow = coefficient.view(1, 1, 1, 1) * template
        loss = photo_loss(source, target, flow)
        optim.zero_grad(); loss.backward(); optim.step()
        if k == 0 or (k+1) % 25 == 0 or k == steps-1:
            history.append({'step': k+1, 'photo_loss': float(loss.item())})
    with torch.no_grad():
        if kind == 'translation':
            final = coefficient.view(1, 2, 1, 1).expand_as(gt).clone()
        else:
            final = coefficient.view(1, 1, 1, 1) * template
    return final, [float(v) for v in coefficient.detach().cpu().numpy()], history


class TinyPairCNN(nn.Module):
    """A 2-image CNN sanity test, not the thesis PairFlowHead."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 24, 5, padding=2), nn.SiLU(),
            nn.Conv2d(24, 24, 5, padding=2), nn.SiLU(),
            nn.Conv2d(24, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 2, 3, padding=1),
        )
        with torch.no_grad():
            self.layers[-1].weight.mul_(.02)
            self.layers[-1].bias.zero_()

    def forward(self, source, target):
        side = source.shape[-1]
        axes = torch.linspace(-1, 1, side, device=source.device)
        yy, xx = torch.meshgrid(axes, axes, indexing='ij')
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(source.shape[0], -1, -1, -1)
        x = torch.cat((source, target, coords), dim=1)
        return 5 * torch.tanh(self.layers(x))


def fit_cnn(source, target, gt, steps, lr, smooth_weight):
    model = TinyPairCNN().to(source.device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for k in range(steps):
        flow = model(source, target)
        photo = photo_loss(source, target, flow)
        smooth = flow_smoothness(flow)
        total = photo + smooth_weight * smooth
        if not torch.isfinite(total):
            raise RuntimeError(f'Nonfinite CNN loss at step {k+1}')
        optim.zero_grad(); total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        if k == 0 or (k+1) % 25 == 0 or k == steps-1:
            history.append({'step':k+1, 'total_loss':float(total.item()),
                            'photo_loss':float(photo.item()), 'smoothness':float(smooth.item())})
    with torch.no_grad():
        final = model(source, target).detach()
    return final, history


def save_panel(path, source, target, gt, learned, title):
    s = source[0,0].detach().cpu().numpy()
    t = target[0,0].detach().cpu().numpy()
    gt_f = gt[0].detach().cpu().numpy()
    pred_f = learned[0].detach().cpu().numpy()
    recon = warp_border(source, learned)[0,0].detach().cpu().numpy()
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    views = [(s, 'Source'), (t, 'Target'), (recon, 'Predicted reconstruction'),
             (gt_f[0], 'True backward flow X'), (pred_f[0], 'Predicted backward flow X'),
             (np.abs(recon-t), 'Abs reconstruction error')]
    for ax, (a, name) in zip(axs.flat, views):
        im = ax.imshow(a, cmap='gray' if 'flow' not in name.lower() else 'coolwarm')
        ax.set_title(name); ax.axis('off')
        if 'flow' in name.lower():
            fig.colorbar(im, ax=ax, shrink=.68)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', default='synthetic_flow_v1')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--side', type=int, default=64)
    p.add_argument('--param-steps', type=int, default=250)
    p.add_argument('--cnn-steps', type=int, default=200)
    p.add_argument('--cnn-lr', type=float, default=0.002)
    p.add_argument('--smooth-weight', type=float, default=.01)
    args = p.parse_args()
    if args.side < 32 or args.param_steps < 1 or args.cnn_steps < 0:
        p.error('Require side>=32, param-steps>0, cnn-steps>=0')
    out = ROOT / 'results' / 'synthetic_flow' / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    device = torch.device(args.device)
    source = make_source(args.side, device)
    gt_flows = {'translation': gt_translation(args.side, device),
                'expansion': gt_expansion(args.side, device)}
    cases = {
        'translation': ('translation', 1.),
        'expansion': ('expansion', 1.),
        'expansion_fading': ('expansion', .65),
    }
    report = {'settings':vars(args), 'methods':{
        'oracle': 'Exact backward warp using the SAME warp_border function as training.',
        'parametric': 'Fit 2 constant flow components or 1 radial scale with photometric loss only.',
        'cnn': 'Overfit one image pair with a small pair-conditioned CNN; not thesis PairFlowHead.'},
        'cases':{}}
    csv_rows = []
    for case, (kind, brightness) in cases.items():
        gt = gt_flows[kind]
        base = warp_border(source, gt).detach()
        target = (base * brightness).detach()
        zero = torch.zeros_like(gt)
        oracle = score(source, target, gt, gt)
        zero_score = score(source, target, zero, gt)
        param_flow, coefficients, param_hist = fit_parametric(
            source, target, gt, kind, args.param_steps,
            lr=.10 if kind == 'translation' else .008)
        param_scores = score(source, target, param_flow, gt)
        case_result = {'zero':zero_score, 'oracle':oracle,
                       'parametric':{**param_scores, 'coefficients':coefficients,
                                     'history':param_hist}}
        save_panel(out / f'{case}_parametric.png', source, target, gt,
                   param_flow, f'{case}: fitted parametric flow')
        for meth, dat in [('zero', zero_score), ('oracle', oracle), ('parametric', param_scores)]:
            csv_rows.append({'case':case, 'method':meth, **dat})
        if args.cnn_steps:
            cnn_flow, cnn_hist = fit_cnn(source, target, gt,
                                        args.cnn_steps, args.cnn_lr, args.smooth_weight)
            cnn_scores = score(source, target, cnn_flow, gt)
            case_result['cnn'] = {**cnn_scores, 'history':cnn_hist}
            csv_rows.append({'case':case, 'method':'cnn', **cnn_scores})
            save_panel(out / f'{case}_cnn.png', source, target, gt,
                       cnn_flow, f'{case}: fitted CNN flow')
        report['cases'][case] = case_result
        print(f'\n{case}:')
        for name in ('zero', 'oracle', 'parametric', 'cnn'):
            if name in case_result:
                m = case_result[name]
                print(f'  {name:>10}: photo={m["photo_loss"]:.6f}, '
                      f'EPE={m["active_epe_px"]:.3f}px, '
                      f'vs_zero={m["improvement_over_zero_pct"]:+.2f}%')
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    with (out / 'metrics.csv').open('w', newline='') as f:
        fields = ['case','method','photo_loss','zero_photo_loss','improvement_over_zero_pct',
                  'active_epe_px','mean_abs_flow_px']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(csv_rows)
    print(f'\nSaved to {out}')
    print('IMPORTANT: Low photometric error alone does not establish accurate flow; inspect active EPE.')

if __name__ == '__main__':
    main()
