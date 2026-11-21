"""SELF-SUPERVISED FlyVis optic-flow training (photometric loss) on synthetic moving edges / textures.

Same network, stimulus pipeline, official DecoderGAVP, activity penalty and optimiser as
scripts/train_edges_supervised.py, but the loss uses NO labels:

    L = mean_valid rho( I_t[p_i] - I_{t-1}[p_i - s * f_i,t] )  +  w_smooth * hex-smoothness(f)

  I_t     BoxEye box-filtered cartesian frame t (391x391): exactly what photoreceptors sample.
  p_i     pixel centre of hexal i (= decoder output column i; verified empirically).
  f_i,t   decoder flow at column i, frame t, in 64x64 px/frame: channel 0 = +x (image column),
          channel 1 = +y (image row), the same convention as the manifest labels.
  s       391/64, the 64 px -> 391 px scale.
  rho     Charbonnier sqrt(x^2 + eps^2).
The flow at frame t is compared with the PREVIOUS frame (t-1 -> t) because the decoder at frame
t has only seen frames <= t (causal). Hexals near the border, or whose source p - s*f leaves
the frame, are masked (BoxEye zero-pads before filtering).

Labels are used ONLY for evaluation metrics (EPE, corr, balanced sign); the best checkpoint is
chosen by validation photometric loss, so the training pipeline is label-free.

From repository root:
    # 1. Sanity check: the photometric loss must be lowest for the ground-truth flow
    python -m scripts.train_edges_selfsupervised --dataset dataset/synthetic_texture_lr --mode check --run-name check
    # 2. Official-recipe settings, self-supervised
    python -m scripts.train_edges_selfsupervised --dataset dataset/synthetic_texture_lr --mode full \
        --network-lr 5e-5 --decoder-lr 5e-5 --activity-penalty 0.1 \
        --steps 30000 --val-samples 100 --test-samples 100 --validate-every 1000 --run-name selfsup_30k
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flyvis.network import Network
from flyvis.task.decoder import DecoderGAVP
from src.flow_training import (activity_penalty, add_activity_penalty_args, add_batching_args, add_perclip,
                               as_jsonable, bn_batch_statistics, central_cells_index, chunks, flow_metrics,
                               initial_state, make_sample, read_manifest, resolve_clips_per_forward, summarize)
from src.flow_training import PERCLIP_KEYS
from src.photoreceptor_mapper import PhotoreceptorWaveMapper

ROOT = Path(__file__).resolve().parent.parent
SCALE = 391 / 64  # 64x64 px displacement -> filtered-frame px


class PhotometricLoss:
    """Label-free warping loss on the hex lattice (see module docstring)."""

    def __init__(self, mapper, network, args, device):
        self.size = 391
        self.p = mapper.pixel_centers(self.size).to(device)          # [721, 2] (y, x)
        self.args = args
        # Hex neighbour pairs for the smoothness term (each undirected pair once).
        df = network.connectome.nodes.to_df().sort_values('index')
        first = df[df['type'].astype(str) == 'R1']
        uv = list(zip(first['u'].to_numpy(), first['v'].to_numpy()))   # column order (= BoxEye order)
        lookup = {k: i for i, k in enumerate(uv)}
        pairs = [(i, lookup[(u + du, v + dv)]) for i, (u, v) in enumerate(uv)
                 for du, dv in ((1, 0), (0, 1), (1, -1)) if (u + du, v + dv) in lookup]
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device)

    def __call__(self, flow, images, first_frame):
        """flow [1, K, 2, 721] for clip frames first_frame..first_frame+K-1; images [T, H, W].
        Returns (loss, photometric term, smoothness term, valid fraction)."""
        k = flow.shape[1]
        f = flow[0]                                                     # [K, 2, 721]
        disp = torch.stack([f[:, 1], f[:, 0]], dim=-1) * SCALE          # [K, 721, (y, x)] px
        q = self.p.unsqueeze(0) - disp                                  # source positions in frame t-1
        lo, hi = self.args.border_margin, self.size - 1 - self.args.border_margin
        inside = lambda z: ((z >= lo) & (z <= hi)).all(dim=-1)
        valid = (inside(self.p).unsqueeze(0) & inside(q)).detach().float()     # [K, 721]
        tgt_frames = images[first_frame:first_frame + k]                # I_t
        src_frames = images[first_frame - 1:first_frame - 1 + k]        # I_{t-1}
        pi = self.p.long()
        target = tgt_frames[:, pi[:, 0], pi[:, 1]]                      # [K, 721] exact pixels
        grid = torch.stack([q[..., 1], q[..., 0]], dim=-1) / (self.size - 1) * 2 - 1   # (x, y) in [-1, 1]
        warped = F.grid_sample(src_frames.unsqueeze(1), grid.unsqueeze(1), mode='bilinear',
                               padding_mode='border', align_corners=True)[:, 0, 0]       # [K, 721]
        diff = target - warped
        rho = torch.sqrt(diff ** 2 + self.args.charbonnier_eps ** 2)
        photo = (rho * valid).sum() / valid.sum().clamp(min=1)
        smooth = (f[:, :, self.pairs[:, 0]] - f[:, :, self.pairs[:, 1]]).abs().mean()
        return photo + self.args.smooth_weight * smooth, photo, smooth, float(valid.mean())


def clip_loss(pred, tgt, images, args, photometric):
    """Photometric loss and metrics of one clip: pred/tgt [1, K, 2, 721], images [T, H, W]."""
    images = images.to(pred.device).float()
    loss, photo, smooth, valid = photometric(pred, images, args.warmup)
    # Reference: photometric loss of zero flow (= plain frame difference) on the same clip.
    with torch.no_grad():
        _, photo_zero, _, _ = photometric(torch.zeros_like(pred), images, args.warmup)
    return loss, dict(photo=float(photo.detach()), photo_zero=float(photo_zero),
                      smooth=float(smooth.detach()), valid_frac=valid, **flow_metrics(pred, tgt))


def decode(decoder, responses, target):
    pred = decoder(responses)
    if pred.shape != target.shape:
        raise RuntimeError(f'Decoder output {tuple(pred.shape)} vs expected {tuple(target.shape)}')
    return pred


def forward(network, decoder, group, args, state, photometric, central_index=None):
    """group: list of B samples (stimulus, target, images) run through the network together
    (state for batch B), so the decoder's train-mode BatchNorm sees all B clips.
    Returns (mean loss over clips, activity penalty or None, list of B per-clip metric dicts)."""
    stimulus = torch.cat([g[0] for g in group])
    target = torch.cat([g[1] for g in group])
    skip = args.preroll + args.warmup
    activity = network(stimulus, dt=args.dt, state=state)
    penalty = activity_penalty(activity, central_index, args) if central_index is not None else None
    tgt = target[:, skip:]
    pred = decode(decoder, activity[:, skip:], tgt)
    losses, metrics = [], []
    for b, (_, _, images) in enumerate(group):
        loss, m = clip_loss(pred[b:b + 1], tgt[b:b + 1], images, args, photometric)
        losses.append(loss); metrics.append(m)
    return torch.stack(losses).mean(), penalty, metrics


def evaluate(network, decoder, examples, args, photometric):
    """One clip at a time. Standard eval mode (BN running statistics) gives the un-prefixed keys;
    the same predictions with BN using each clip's own statistics give perclip_bn_* (CLAUDE.md 6.13)."""
    network.eval(); decoder.eval()
    skip = args.preroll + args.warmup
    metrics, perclip = [], []
    with torch.no_grad():
        for stimulus, target, images in examples:
            responses = network(stimulus, dt=args.dt, state=initial_state(network, args))[:, skip:]
            tgt = target[:, skip:]
            metrics.append(clip_loss(decode(decoder, responses, tgt), tgt, images, args, photometric)[1])
            with bn_batch_statistics(decoder):
                perclip.append(clip_loss(decode(decoder, responses, tgt), tgt, images, args, photometric)[1])
    network.train(); decoder.train()
    out, pc = summarize(metrics), summarize(perclip)
    for o in (out, pc):
        o['photo_improvement_over_zero_pct'] = 100 * (1 - o['photo'] / o['photo_zero'])
    return add_perclip(out, pc, keys=('photo', 'photo_improvement_over_zero_pct') + PERCLIP_KEYS)


@torch.no_grad()
def run_check(examples, args, photometric, out):
    """Photometric loss for candidate constant flows. The ground truth must be lowest,
    otherwise coordinates, sign or scale are wrong."""
    candidates = {'ground_truth': lambda t: t, 'zero': lambda t: 0 * t, 'negated': lambda t: -t,
                  'axes_swapped': lambda t: t.flip(2), 'half': lambda t: .5 * t, 'one_and_half': lambda t: 1.5 * t}
    skip = args.preroll + args.warmup
    res = {name: [] for name in candidates}
    for stimulus, target, images in examples:
        tgt = target[:, skip:]
        for name, fn in candidates.items():
            res[name].append(float(photometric(fn(tgt), images.to(tgt.device).float(), args.warmup)[1]))
    summary = {name: float(np.mean(v)) for name, v in res.items()}
    summary['ground_truth_lowest_on_fraction_of_clips'] = float(np.mean(
        [min(res, key=lambda n: res[n][i]) == 'ground_truth' for i in range(len(examples))]))
    (out / 'check_metrics.json').write_text(json.dumps(summary, indent=2))
    print('Photometric loss by candidate flow:', json.dumps(summary, indent=2), flush=True)
    ok = min(candidates, key=lambda n: summary[n]) == 'ground_truth'
    print('CHECK PASSED: ground truth has the lowest photometric loss.' if ok else
          'CHECK FAILED: ground truth is not the lowest; fix coordinates/sign/scale before training.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('overfit', 'full', 'check'), default='full')
    p.add_argument('--dataset', type=Path, default=ROOT / 'dataset' / 'synthetic_texture_lr')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4, help='clips per optimizer step (grad accumulation)')
    p.add_argument('--overfit-samples', type=int, default=8)
    p.add_argument('--val-samples', type=int, default=100)
    p.add_argument('--test-samples', type=int, default=100)
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5, help='>= 1: the first scored frame needs a previous frame')
    p.add_argument('--preroll', type=int, default=10)
    p.add_argument('--dt', type=float, default=.02)
    p.add_argument('--t-pre', type=float, default=.5)
    p.add_argument('--network-lr', type=float, default=5e-5)
    p.add_argument('--decoder-lr', type=float, default=5e-5)
    p.add_argument('--decoder-dropout', type=float, default=0.5)
    p.add_argument('--no-decoder-bn', action='store_true')
    p.add_argument('--charbonnier-eps', type=float, default=1e-3)
    p.add_argument('--smooth-weight', type=float, default=0.0, help='first-order hex smoothness weight')
    p.add_argument('--border-margin', type=float, default=10.0, help='masked border width, filtered-frame px')
    add_activity_penalty_args(p)
    add_batching_args(p)
    p.add_argument('--load-checkpoint', type=Path, default=None)
    p.add_argument('--freeze-network', action='store_true')
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--run-name', default='selfsupervised_experiment')
    args = p.parse_args()
    resolve_clips_per_forward(p, args)
    if min(args.steps, args.val_samples, args.test_samples, args.batch_size) < 1:
        p.error('steps, batch-size, val-samples, test-samples must be positive')
    if not 1 <= args.warmup < args.frames:
        p.error('warmup must be between 1 and frames-1')
    if '/' in args.run_name or '\\' in args.run_name or args.run_name in ('', '.', '..'):
        p.error('run-name must be a simple folder name')
    config = json.loads((args.dataset / 'config.json').read_text())
    if config['frames'] != args.frames or config['side'] != 64:
        p.error('Dataset must have --frames frames at 64x64')
    records = read_manifest(args.dataset)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    if len(records['val']) < args.val_samples or len(records['test']) < args.test_samples:
        p.error('Too few validation or test examples')

    out = ROOT / 'results' / args.dataset.name / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f).exists() for f in ('history.csv', 'final.pt', 'check_metrics.json')):
        raise FileExistsError(f'Run already exists: {out}; use a new --run-name')
    device = torch.device(args.device)
    network = Network()
    base = torch.load(ROOT / 'checkpoints' / 'flyvis_initial.pt', map_location='cpu', weights_only=True)
    network.load_state_dict(base['network'], strict=True)
    decoder = DecoderGAVP(network.connectome, shape=[8, 2], kernel_size=5,
                          p_dropout=args.decoder_dropout, batch_norm=not args.no_decoder_bn)
    mapper = PhotoreceptorWaveMapper(network)
    if args.load_checkpoint is not None:
        ck = torch.load(args.load_checkpoint, map_location='cpu', weights_only=True)
        network.load_state_dict(ck['network'], strict=True)
        decoder.load_state_dict(ck['decoder'], strict=True)
        print(f'Loaded weights from {args.load_checkpoint} (step {ck.get("step")})', flush=True)
    network, decoder = network.to(device), decoder.to(device)
    (out / 'settings.json').write_text(json.dumps(as_jsonable(vars(args)), indent=2))
    photometric = PhotometricLoss(mapper, network, args, device)

    def load(row):
        stimulus, target, images = make_sample(args.dataset, row, mapper, device, args.frames,
                                               args.preroll, with_images=True)
        return stimulus, target, images.half().cpu()   # images cached on CPU in fp16

    val_rows = rng.sample(records['val'], args.val_samples)
    test_rows = rng.sample(records['test'], args.test_samples)
    if args.mode == 'check':
        run_check([load(r) for r in val_rows], args, photometric, out)
        return

    if args.freeze_network:
        for param in network.parameters():
            param.requires_grad_(False)
    net_parameters = [v for v in network.parameters() if v.requires_grad]
    optim_groups = [{'params': list(decoder.parameters()), 'lr': args.decoder_lr}]
    if net_parameters:
        optim_groups.append({'params': net_parameters, 'lr': args.network_lr})
    optimizer = torch.optim.Adam(optim_groups)
    initial_params = [x.detach().clone() for x in net_parameters]
    central_index = (central_cells_index(network, device)
                     if args.activity_penalty > 0 and not args.freeze_network else None)
    train_rows = rng.sample(records['train'], args.overfit_samples) if args.mode == 'overfit' else records['train']
    training_cached = [load(r) for r in train_rows] if args.mode == 'overfit' else None
    validation_cached = [load(r) for r in val_rows]
    test_cached = [load(r) for r in test_rows]
    config_out = as_jsonable(vars(args))
    (out / 'sample_ids.json').write_text(json.dumps({
        'train_overfit': [r['id'] for r in train_rows] if args.mode == 'overfit' else None,
        'val': [r['id'] for r in val_rows], 'test': [r['id'] for r in test_rows]}, indent=2))

    def save(filename, step):
        torch.save({'network': network.state_dict(), 'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(), 'step': step,
                    'type': 'official_DecoderGAVP_selfsupervised_photometric', 'settings': config_out},
                   out / filename)

    print(f'Experiment: self-supervised {args.mode}; FlyVis frozen={args.freeze_network}; batch={args.batch_size}; '
          f'train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}', flush=True)
    print('Loss uses NO labels (photometric warping); labels only for evaluation.', flush=True)
    history_fields = ['step', 'id', 'loss', 'photo', 'photo_zero', 'smooth', 'valid_frac', 'epe_px', 'zero_epe_px',
                      'improvement_over_zero_pct', 'flow_abs_px', 'sign_accuracy', 'pred_std', 'pred_mean_x',
                      'pred_mean_y', 'network_change_l2', 'network_grad_norm', 'activity_penalty']
    best = float('inf')
    with (out / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history_fields, extrasaction='ignore')
        writer.writeheader()
        for step in range(1, args.steps + 1):
            network.train(); decoder.train()
            optimizer.zero_grad(set_to_none=True)
            batch_metrics, batch_losses, batch_ids = [], [], []
            use_pen = central_index is not None and step <= args.activity_penalty_stop
            pen_grad, pen_values = None, []
            drawn = []
            for _ in range(args.batch_size):
                if training_cached is None:
                    row = train_rows[rng.randrange(len(train_rows))]
                    drawn.append((row, load(row)))
                else:
                    index = rng.randrange(len(training_cached))
                    drawn.append((train_rows[index], training_cached[index]))
            for group in chunks(drawn, args.clips_per_forward):
                weight = len(group) / args.batch_size
                state = initial_state(network, args, len(group))
                loss, pen, ms = forward(network, decoder, [sample for _, sample in group], args, state,
                                        photometric, central_index if use_pen else None)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                if pen is not None:   # gradient for nodes_bias only, kept apart from the photometric gradient
                    g, = torch.autograd.grad(pen * weight, network.nodes_bias, retain_graph=True)
                    pen_grad = g if pen_grad is None else pen_grad + g
                    pen_values.append(float(pen.detach()))
                (loss * weight).backward()
                batch_metrics.extend(ms)
                batch_losses.extend(m['photo'] + args.smooth_weight * m['smooth'] for m in ms)
                batch_ids.extend(str(row['id']) for row, _ in group)
            grad_norm = math.sqrt(sum(float(x.grad.detach().square().sum()) for x in net_parameters
                                      if x.grad is not None)) if net_parameters else 0.0
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            if net_parameters:
                torch.nn.utils.clip_grad_norm_(net_parameters, 1.0)
            optimizer.step()
            if pen_grad is not None:   # official: separate SGD step at the network lr, then clamp
                with torch.no_grad():
                    network.nodes_bias.sub_(args.network_lr * pen_grad)
                network.clamp()
            delta = math.sqrt(sum(float((x.detach() - orig).square().sum())
                                  for x, orig in zip(net_parameters, initial_params))) if net_parameters else 0.0
            metrics = {k: float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]}
            metrics.update(step=step, id='|'.join(batch_ids), loss=float(np.mean(batch_losses)),
                           improvement_over_zero_pct=100 * (1 - metrics['epe_px'] / metrics['zero_epe_px']),
                           network_change_l2=delta, network_grad_norm=grad_norm,
                           activity_penalty=float(np.mean(pen_values)) if pen_values else 0.0)
            writer.writerow(metrics)
            if step == 1 or step % 25 == 0:
                print('Train:', json.dumps(metrics), flush=True)
            if step % args.validate_every == 0 or step == args.steps:
                val = evaluate(network, decoder, validation_cached, args, photometric)
                val['step'] = step
                (out / f'validation_step_{step:06d}.json').write_text(json.dumps(val, indent=2))
                print('Validation:', json.dumps(val), flush=True)
                if val['photo'] < best:   # label-free model selection
                    best = val['photo']
                    save('best.pt', step)
                    (out / 'best_metrics.json').write_text(json.dumps(val, indent=2))
            f.flush()
    save('final.pt', args.steps)
    final_test = evaluate(network, decoder, test_cached, args, photometric)
    final_test['step'] = args.steps
    (out / 'test_metrics.json').write_text(json.dumps(final_test, indent=2))
    print('Test:', json.dumps(final_test), flush=True)
    print(f'Finished: {out}; best validation photometric loss {best:.6f}', flush=True)


if __name__ == '__main__':
    main()
