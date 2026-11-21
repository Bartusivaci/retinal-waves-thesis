"""Helpers shared by scripts/train_edges_supervised.py and scripts/train_edges_selfsupervised.py.

Data loading, initial state, the official FlyVis activity penalty, and metrics a constant
predictor cannot fake. Keep this module free of training-loop logic.
"""
from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_COLUMNS = 721


def as_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return obj


def read_manifest(root: Path):
    with (root / 'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    return {split: [r for r in rows if r['split'] == split]
            for split in ('train', 'val', 'test')}


def make_sample(root, record, mapper, device, n_frames, preroll=0, with_images=False):
    """Returns stimulus [1, preroll+T, n_neurons] and target [1, preroll+T, 2, 721]
    (+ box-filtered cartesian frames [T, 391, 391] if with_images).
    Pre-roll frames repeat frame 0 and carry a zero target; they are never scored.
    The target is only used for supervision or evaluation; it is read from the manifest."""
    with np.load(root / record['filename']) as data:
        frames = np.asarray(data['frames']).copy()  # T,H,W uint8
    if frames.ndim != 3 or frames.shape[0] != n_frames:
        raise ValueError(f'Unexpected frames {frames.shape}; expected {n_frames} frames')
    movie = torch.from_numpy(frames).float().unsqueeze(1) / 255.0
    rendered_cart = F.interpolate(movie, size=(256, 256), mode='bilinear', align_corners=False)
    stimulus = mapper(rendered_cart).to(device).clone()  # [1,T,n_neurons]
    assert stimulus.ndim == 3 and stimulus.shape[0:2] == (1, n_frames)
    if preroll > 0:
        pad = stimulus[:, :1].expand(-1, preroll, -1)
        stimulus = torch.cat([pad, stimulus], dim=1).contiguous()
    total = n_frames + preroll
    # Forward flow, same convention as the Sintel tutorial. Do NOT negate.
    velocity = [float(record['dx_px_per_frame']), float(record['dy_px_per_frame'])]
    target = torch.tensor(velocity, device=device, dtype=stimulus.dtype)
    target = target.view(1, 1, 2, 1).expand(1, total, 2, N_COLUMNS).contiguous()
    if with_images:
        return stimulus, target, mapper.filtered(rendered_cart).to(device)
    return stimulus, target


def initial_state(network, args, batch=1):
    if args.t_pre <= 0:
        return None
    with torch.no_grad():
        return network.steady_state(args.t_pre, args.dt, batch)


def chunks(items, size):
    """Split a list into consecutive chunks of at most `size` items."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def add_batching_args(p):
    p.add_argument('--clips-per-forward', type=int, default=0,
                   help='clips run through the network together in one forward (0 = --batch-size, as in the '
                        'official recipe). Decoder BatchNorm statistics are computed over these clips in training; '
                        'with 1, BN normalises each clip by itself and eval mode no longer matches (CLAUDE.md 6.13)')


def resolve_clips_per_forward(p, args):
    if args.clips_per_forward == 0:
        args.clips_per_forward = args.batch_size
    if args.clips_per_forward < 1 or args.batch_size % args.clips_per_forward:
        p.error('--clips-per-forward must divide --batch-size')


@contextmanager
def bn_batch_statistics(module):
    """Inside this block, BatchNorm layers normalise with the statistics of the current input
    (as in training) instead of their running statistics; running statistics are restored afterwards.
    Other layers (dropout) keep their mode, so use it on a module in eval mode."""
    bns = [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [(m.training, {k: v.clone() for k, v in m.named_buffers()}) for m in bns]
    for m in bns:
        m.train()
    try:
        yield
    finally:
        with torch.no_grad():
            for m, (training, buffers) in zip(bns, saved):
                m.train(training)
                for k, v in m.named_buffers():
                    v.copy_(buffers[k])


PERCLIP_KEYS = ('epe_px', 'improvement_over_zero_pct', 'pred_mean_x', 'corr_clipmean_x', 'corr_clipmean_y',
                'balanced_sign_acc_x', 'balanced_sign_acc_y', 'pred_between_clip_std_x')


def add_perclip(summary, perclip_summary, keys=PERCLIP_KEYS):
    """Add the per-clip-BN evaluation (decoder BN uses each clip's own statistics) as perclip_bn_* keys.
    The un-prefixed keys stay the standard eval-mode (running statistics) result."""
    for k in keys:
        if k in perclip_summary:
            summary[f'perclip_bn_{k}'] = perclip_summary[k]
    return summary


def central_cells_index(network, device):
    return torch.as_tensor(np.asarray(network.connectome.central_cells_index[:]),
                           dtype=torch.long, device=device)


def activity_penalty(activity, central_index, args):
    """Official FlyVis activity penalty (flyvis.solver.Penalty.activity_penalty_step):
    pushes the temporal-mean activity of the central cell of each type toward a baseline,
    weighting below-baseline deviations more. Applied to nodes_bias only (see training loops)."""
    n_frames = activity.shape[1]
    mean = activity[:, n_frames // 4:, central_index].mean(dim=1)
    x = args.activity_baseline - mean
    x = torch.where(x > 0, x * args.below_baseline_weight, x * args.above_baseline_weight)
    return args.activity_penalty * (x ** 2).mean()


def add_activity_penalty_args(p):
    p.add_argument('--activity-penalty', type=float, default=0.0,
                   help='official FlyVis activity penalty coefficient on nodes_bias (official 0.1; 0 = off)')
    p.add_argument('--activity-baseline', type=float, default=5.0)
    p.add_argument('--below-baseline-weight', type=float, default=1.0)
    p.add_argument('--above-baseline-weight', type=float, default=0.1)
    p.add_argument('--activity-penalty-stop', type=int, default=150000, help='last step with the penalty (official 150k)')


def flow_metrics(pred, tgt):
    """Per-clip metrics of predicted flow [1,T,2,721] against the (constant) label."""
    pred, tgt = pred.detach(), tgt.detach()
    endpoint_error = (pred - tgt).square().sum(dim=2).sqrt().mean()
    zero_epe = tgt.square().sum(dim=2).sqrt().mean()
    mask = tgt.abs() > .2
    sign_accuracy = ((pred.sign() == tgt.sign()) & mask).sum() / mask.sum().clamp(min=1)
    pm = pred.mean(dim=(0, 1, 3))
    return dict(epe_px=float(endpoint_error), zero_epe_px=float(zero_epe),
                flow_abs_px=float(pred.abs().mean()), sign_accuracy=float(sign_accuracy),
                pred_std=float(pred.std()), pred_mean_x=float(pm[0]), pred_mean_y=float(pm[1]),
                label_x=float(tgt[0, 0, 0, 0]), label_y=float(tgt[0, 0, 1, 0]))


def input_dependence(metrics):
    """Metrics a constant predictor cannot fake (EPE/sign accuracy can, on an imbalanced
    subset: lr_task scored +37.7% on a 74%-leftward test subset with a constant output)."""
    out = {}
    for ax in ('x', 'y'):
        p = np.array([m[f'pred_mean_{ax}'] for m in metrics])
        y = np.array([m[f'label_{ax}'] for m in metrics])
        out[f'corr_clipmean_{ax}'] = float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 and y.std() > 0 else 0.0
        out[f'pred_between_clip_std_{ax}'] = float(p.std())
        moving = np.abs(y) > .2
        accs = [float((np.sign(p[moving & (y * s > 0)]) == s).mean())
                for s in (-1, 1) if (moving & (y * s > 0)).any()]
        out[f'balanced_sign_acc_{ax}'] = float(np.mean(accs)) if len(accs) == 2 else float('nan')
    return out


def summarize(metrics):
    """Average per-clip metric dicts (skipping label_*) and add input-dependence metrics."""
    means = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]
             if not key.startswith('label_')}
    means['count'] = len(metrics)
    means['improvement_over_zero_pct'] = 100.0 * (1.0 - means['epe_px'] / means['zero_epe_px'])
    means.update(input_dependence(metrics))
    return means
