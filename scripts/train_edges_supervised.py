"""SUPERVISED FlyVis optic-flow training on synthetic moving edges / textures.

Mirrors the FlyVis Sintel tutorial: official DecoderGAVP, steady-state initialisation,
supervised EPE against the known (constant) forward flow of each clip. Network trained
from scratch (checkpoints/flyvis_initial.pt). Self-supervised counterpart:
scripts/train_edges_selfsupervised.py. Shared helpers: src/flow_training.py.

Official recipe (the setup that learned left/right direction, results/synthetic_texture_lr/official_recipe_30k):
    python -m scripts.train_edges_supervised --dataset dataset/synthetic_texture_lr --mode full \
        --input-norm none --network-lr 5e-5 --decoder-lr 5e-5 --activity-penalty 0.1 \
        --steps 30000 --val-samples 100 --test-samples 100 --validate-every 1000 --run-name <name>

Other modes: overfit (few cached clips), probe (ridge on frozen responses), diagnose (what the
decoder receives). Results folder: results/<dataset folder name>/<run-name>.
Labels are *forward* displacement in 64x64 pixels per generated frame, constant over all
721 columns. One generated frame = dt 0.02 s.
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
from flyvis.task.objectives import epe
from src.flow_training import (activity_penalty, add_activity_penalty_args, add_batching_args, add_perclip,
                               as_jsonable, bn_batch_statistics, central_cells_index, chunks, flow_metrics,
                               initial_state, make_sample, read_manifest, resolve_clips_per_forward, summarize)
from src.photoreceptor_mapper import PhotoreceptorWaveMapper

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# Input normalisation (--input-norm zscore). Only useful for diagnosing the untrained
# network (whose output cells move by ~1e-4); use --input-norm none for unfrozen training,
# where the frozen 1/sd (~1e4) amplifies every network step (see CLAUDE.md 6.11).
# ----------------------------------------------------------------------------
NORM = {'mu': None, 'sd': None, 'offset': 3.0}


def apply_norm(responses):
    if NORM['mu'] is None:
        return responses
    return NORM['offset'] + (responses - NORM['mu']) / NORM['sd']


@torch.no_grad()
def calibrate_norm(network, mapper, rows, args, device):
    network.eval()
    skip = args.preroll + args.warmup
    n, s1, s2 = 0, None, None
    for row in rows:
        stimulus, _ = make_sample(args.dataset, row, mapper, device, args.frames, args.preroll)
        state = initial_state(network, args)
        r = network(stimulus, dt=args.dt, state=state)[0, skip:].double()   # [T', N]
        s1 = r.sum(0) if s1 is None else s1 + r.sum(0)
        s2 = (r ** 2).sum(0) if s2 is None else s2 + (r ** 2).sum(0)
        n += r.shape[0]
    mu = s1 / n
    sd = (s2 / n - mu ** 2).clamp(min=0).sqrt()
    at_floor = float((sd < args.norm_floor).float().mean())
    sd = sd.clamp(min=args.norm_floor)
    NORM['mu'] = mu.float().view(1, 1, -1)
    NORM['sd'] = sd.float().view(1, 1, -1)
    NORM['offset'] = args.norm_offset
    print(f'Input norm calibrated on {len(rows)} clips: median node std {float(sd.median()):.2e}, '
          f'{100 * at_floor:.1f}% of nodes at std floor {args.norm_floor:g}', flush=True)


def decode(decoder, responses, target):
    pred = decoder(responses)
    if pred.shape != target.shape:
        raise RuntimeError(f'Decoder output {tuple(pred.shape)} vs target {tuple(target.shape)}. '
                           'Expected [B,T,2,721]; check installed FlyVis version.')
    return pred


def forward(network, decoder, stimulus, target, args, state, central_index=None):
    """stimulus [B, T, N] and target [B, T, 2, 721] for B clips run together (state for batch B).
    Returns (mean EPE, activity penalty or None, list of B per-clip metric dicts)."""
    # Stimulus is already a full-network input; do NOT add_input / BoxEye again.
    # Pre-roll and warmup frames only drive the network's recurrent state. They are cut BEFORE
    # the decoder, so its BatchNorm (which in train mode normalises over the clips and frames of
    # the forward) only sees scored frames.
    skip = args.preroll + args.warmup
    activity = network(stimulus, dt=args.dt, state=state)
    penalty = activity_penalty(activity, central_index, args) if central_index is not None else None
    tgt = target[:, skip:]
    pred = decode(decoder, apply_norm(activity[:, skip:]), tgt)
    metrics = [flow_metrics(pred[b:b + 1], tgt[b:b + 1]) for b in range(pred.shape[0])]
    return epe(pred, tgt), penalty, metrics


def evaluate(network, decoder, examples, args, device):
    """One clip at a time. Standard eval mode (BN running statistics) gives the un-prefixed keys;
    the same predictions with BN using each clip's own statistics give perclip_bn_* (CLAUDE.md 6.13)."""
    network.eval(); decoder.eval()
    skip = args.preroll + args.warmup
    metrics, perclip = [], []
    with torch.no_grad():
        for stimulus, target in examples:
            responses = apply_norm(network(stimulus, dt=args.dt, state=initial_state(network, args))[:, skip:])
            tgt = target[:, skip:]
            metrics.append(flow_metrics(decode(decoder, responses, tgt), tgt))
            with bn_batch_statistics(decoder):
                perclip.append(flow_metrics(decode(decoder, responses, tgt), tgt))
    network.train(); decoder.train()
    return add_perclip(summarize(metrics), summarize(perclip))


# ----------------------------------------------------------------------------
# Linear probe: can ANY linear readout of the (frozen) network responses predict
# velocity? Features = per-neuron temporal mean and std over scored frames.
# ----------------------------------------------------------------------------
def type_indices(network, names, device):
    """Indices of network nodes whose cell type is in `names` (comma-separated string)."""
    wanted = {n.strip() for n in names.split(',') if n.strip()}
    raw = network.connectome.nodes.type[:]
    types = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in raw])
    keep = np.nonzero(np.isin(types, list(wanted)))[0]
    if len(keep) == 0:
        raise ValueError(f'No nodes match {sorted(wanted)}; available e.g. {sorted(set(types))[:20]}')
    print(f'Probe restricted to {len(keep)} nodes of types {sorted(wanted)}', flush=True)
    return torch.from_numpy(keep).to(device)


@torch.no_grad()
def probe_features(network, mapper, rows, args, device, keep=None):
    """--probe-features: time_stats (mean+std over time), time_mean, last_frame (one frame,
    no temporal aggregation), input (raw photoreceptor drive, network bypassed = control)."""
    network.eval()
    feats, labels = [], []
    skip = args.preroll + args.warmup
    for row in rows:
        stimulus, target = make_sample(args.dataset, row, mapper, device, args.frames, args.preroll)
        if args.probe_features == 'input':
            r = stimulus[:, skip:]
        else:
            state = initial_state(network, args)
            r = network(stimulus, dt=args.dt, state=state)[:, skip:]  # [1,T',N]
        if keep is not None:
            r = r[..., keep]
        if args.probe_features == 'last_frame':
            f = r[:, -1]
        elif args.probe_features == 'time_mean':
            f = r.mean(1)
        else:
            f = torch.cat([r.mean(1), r.std(1)], dim=-1)
        feats.append(f[0].float().cpu())
        labels.append(target[0, 0, :, 0].float().cpu())
    return torch.stack(feats), torch.stack(labels)


@torch.no_grad()
def run_diagnose(network, decoder, mapper, records, args, device, out):
    """What does the decoder actually receive, and does it vary between clips?
    Mirrors DecoderGAVP.forward: relu of the output-cell-type responses."""
    network.eval(); decoder.eval()
    rows = random.Random(args.seed).sample(records['val'], args.val_samples)
    skip = args.preroll + args.warmup
    xs, preds, labels = [], [], []
    for row in rows:
        stimulus, target = make_sample(args.dataset, row, mapper, device, args.frames, args.preroll)
        state = initial_state(network, args)
        responses = apply_norm(network(stimulus, dt=args.dt, state=state)[:, skip:])
        decoder.dvs_channels.update(responses)
        xs.append(F.relu(decoder.dvs_channels.output)[0].float().cpu())      # [T,C,721]
        preds.append(decoder(responses)[0].float().cpu())                    # [T,2,721]
        labels.append(target[0, 0, :, 0].float().cpu())
    X, P, Y = torch.stack(xs), torch.stack(preds), torch.stack(labels)       # [n,T,C,H] ...
    frac_pos_channel = (X > 0).float().mean(dim=(0, 1, 3))
    clip_mean = X.mean(1)                                                     # [n,C,H]
    between = clip_mean.std(0).mean().item()          # variation of time-averaged input across clips
    within_time = X.std(1).mean().item()              # variation over time inside a clip
    pm = P.mean(dim=(1, 3))                                                   # [n,2] clip-mean output
    out_between = pm.std(0).tolist()
    out_within = P.std(dim=(1, 3)).mean(0).tolist()
    corr = [float(np.corrcoef(pm[:, i].numpy(), Y[:, i].numpy())[0, 1]) for i in range(2)]
    res = {'n_output_channels': int(X.shape[2]),
           'frac_positive_input_overall': float((X > 0).float().mean()),
           'frac_positive_per_channel': [round(float(v), 3) for v in frac_pos_channel],
           'dead_channels': int((frac_pos_channel == 0).sum()),
           'input_mean': float(X.mean()),
           'input_between_clip_std': between, 'input_within_clip_time_std': within_time,
           'output_between_clip_std_xy': out_between, 'output_within_clip_std_xy': out_within,
           'corr_clipmean_output_vs_label_xy': corr}
    (out / 'diagnose_metrics.json').write_text(json.dumps(res, indent=2))
    print('Diagnose:', json.dumps(res, indent=2), flush=True)


def run_probe(network, mapper, records, args, device, out):
    rng = random.Random(args.seed)
    train_rows = rng.sample(records['train'], args.probe_train)
    val_rows = rng.sample(records['val'], args.val_samples)
    test_rows = rng.sample(records['test'], args.test_samples)
    keep = type_indices(network, args.probe_types, device) if args.probe_types else None
    Xtr, Ytr = probe_features(network, mapper, train_rows, args, device, keep)
    Xva, Yva = probe_features(network, mapper, val_rows, args, device, keep)
    Xte, Yte = probe_features(network, mapper, test_rows, args, device, keep)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva, Xte = [((x - mu) / sd).double() for x in (Xtr, Xva, Xte)]
    ymean = Ytr.mean(0).double()
    Ytr, Yva, Yte = Ytr.double(), Yva.double(), Yte.double()
    D = Xtr.shape[1]
    Ktr = Xtr @ Xtr.T / D

    def predict(X, lam):  # dual ridge regression
        eye = torch.eye(len(Ktr), dtype=torch.float64, device=Ktr.device)
        alpha = torch.linalg.solve(Ktr + lam * eye, Ytr - ymean)
        return (X @ Xtr.T / D) @ alpha + ymean

    def score(pred, Y):
        epe_v = (pred - Y).norm(dim=1).mean().item()
        zero = Y.norm(dim=1).mean().item()
        r2 = (1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).tolist()
        sign = ((pred.sign() == Y.sign()).double().mean(0)).tolist()
        return dict(epe_px=epe_v, zero_epe_px=zero, improvement_over_zero_pct=100 * (1 - epe_v / zero),
                    r2_dx=r2[0], r2_dy=r2[1], sign_acc_dx=sign[0], sign_acc_dy=sign[1])

    best_lam, best = None, None
    for lam in (1e-3, 1e-2, 1e-1, 1, 10):
        s = score(predict(Xva, lam), Yva)
        print(f'probe lam={lam}: val {json.dumps(s)}', flush=True)
        if best is None or s['epe_px'] < best['epe_px']:
            best_lam, best = lam, s
    test = score(predict(Xte, best_lam), Yte)
    result = {'lambda': best_lam, 'val': best, 'test': test, 'n_train': len(Xtr), 'n_features': D,
              'features': args.probe_features, 'types': args.probe_types or 'all',
              'dataset': str(args.dataset)}
    (out / 'probe_metrics.json').write_text(json.dumps(result, indent=2))
    print('Probe result:', json.dumps(result), flush=True)
    print('Interpretation: R2 > 0 and sign accuracy >> 0.5 means velocity is linearly decodable '
          'from the network responses; otherwise fix data/network first.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('overfit', 'full', 'probe', 'diagnose'), default='overfit')
    p.add_argument('--dataset', type=Path, default=ROOT / 'dataset' / 'synthetic_moving_edges_v2')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4, help='clips per optimizer step (grad accumulation)')
    p.add_argument('--overfit-samples', type=int, default=4)
    p.add_argument('--probe-train', type=int, default=200)
    p.add_argument('--probe-features', choices=('time_stats', 'time_mean', 'last_frame', 'input'),
                   default='time_stats')
    p.add_argument('--probe-types', default='',
                   help='comma-separated cell types to keep, e.g. T4a,T4b,T4c,T4d,T5a,T5b,T5c,T5d')
    p.add_argument('--val-samples', type=int, default=12)
    p.add_argument('--test-samples', type=int, default=12)
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--preroll', type=int, default=10,
                   help='static copies of frame 0 prepended to each clip (not scored)')
    p.add_argument('--dt', type=float, default=.02)
    p.add_argument('--t-pre', type=float, default=.5)
    p.add_argument('--network-lr', type=float, default=1e-5)
    p.add_argument('--decoder-lr', type=float, default=1e-3)
    p.add_argument('--decoder-dropout', type=float, default=0.5,
                   help='DecoderGAVP p_dropout (official default 0.5; use 0 for overfit diagnostics)')
    p.add_argument('--no-decoder-bn', action='store_true', help='disable DecoderGAVP BatchNorm')
    p.add_argument('--input-norm', choices=('zscore', 'none'), default='zscore',
                   help='standardise network responses with fixed calibration stats before the decoder')
    p.add_argument('--norm-clips', type=int, default=40, help='training clips used for calibration')
    p.add_argument('--norm-offset', type=float, default=3.0)
    p.add_argument('--norm-floor', type=float, default=1e-5, help='minimum per-node std')
    add_activity_penalty_args(p)
    add_batching_args(p)
    p.add_argument('--load-checkpoint', type=Path, default=None,
                   help='load network+decoder weights from a best.pt/final.pt (optimizer not restored)')
    p.add_argument('--freeze-network', action='store_true',
                   help='Train only official decoder; FlyVis baseline remains fixed')
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--run-name', default='supervised_experiment')
    args = p.parse_args()
    resolve_clips_per_forward(p, args)
    if args.steps < 1 or args.val_samples < 1 or args.test_samples < 1 or args.batch_size < 1:
        p.error('steps, batch-size, val-samples, test-samples must be positive')
    if args.preroll < 0:
        p.error('preroll must be >= 0')
    if not 0 <= args.warmup < args.frames:
        p.error('warmup must be between 0 and frames-1')
    if '/' in args.run_name or '\\' in args.run_name or args.run_name in ('', '.', '..'):
        p.error('run-name must be a simple folder name')
    config = json.loads((args.dataset / 'config.json').read_text())
    if config['frames'] != args.frames or config['side'] != 64:
        p.error('Dataset must have --frames frames at 64x64')
    records = read_manifest(args.dataset)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    need_train = args.probe_train if args.mode == 'probe' else args.overfit_samples
    if len(records['train']) < need_train or len(records['val']) < args.val_samples:
        p.error('Too few training or validation examples')
    if len(records['test']) < args.test_samples:
        p.error('Too few test examples')

    out = ROOT / 'results' / args.dataset.name / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'history.csv').exists() or (out / 'final.pt').exists() or (out / 'probe_metrics.json').exists() \
            or (out / 'diagnose_metrics.json').exists():
        raise FileExistsError(f'Run already exists: {out}; use a new --run-name')
    device = torch.device(args.device)
    network = Network()
    checkpoint = torch.load(ROOT / 'checkpoints' / 'flyvis_initial.pt', map_location='cpu', weights_only=True)
    network.load_state_dict(checkpoint['network'], strict=True)
    decoder = DecoderGAVP(network.connectome, shape=[8, 2], kernel_size=5,
                          p_dropout=args.decoder_dropout, batch_norm=not args.no_decoder_bn)
    mapper = PhotoreceptorWaveMapper(network)
    if args.load_checkpoint is not None:
        ck = torch.load(args.load_checkpoint, map_location='cpu', weights_only=True)
        network.load_state_dict(ck['network'], strict=True)
        decoder.load_state_dict(ck['decoder'], strict=True)
        print(f'Loaded weights from {args.load_checkpoint} (step {ck.get("step")})', flush=True)
        if args.input_norm == 'zscore' and ck.get('input_norm') is not None:
            NORM.update({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ck['input_norm'].items()})
            print('Using input-norm statistics stored in the checkpoint.', flush=True)
    network = network.to(device)
    decoder = decoder.to(device)
    (out / 'settings.json').write_text(json.dumps(as_jsonable(vars(args)), indent=2))

    if args.mode == 'probe':
        run_probe(network, mapper, records, args, device, out)
        return
    if args.input_norm == 'zscore' and NORM['mu'] is None:
        calib_rows = random.Random(args.seed + 1).sample(records['train'], min(args.norm_clips, len(records['train'])))
        calibrate_norm(network, mapper, calib_rows, args, device)
    if args.mode == 'diagnose':
        run_diagnose(network, decoder, mapper, records, args, device, out)
        return

    print('Official DecoderGAVP parameters:', sum(v.numel() for v in decoder.parameters()), flush=True)
    print('FlyVis trainable scalar entries:', sum(v.numel() for v in network.parameters() if v.requires_grad), flush=True)
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
    val_rows = rng.sample(records['val'], args.val_samples)
    test_rows = rng.sample(records['test'], args.test_samples)

    def load(row):
        return make_sample(args.dataset, row, mapper, device, args.frames, args.preroll)

    training_cached = [load(row) for row in train_rows] if args.mode == 'overfit' else None
    validation_cached = [load(row) for row in val_rows]
    test_cached = [load(row) for row in test_rows]
    config_out = as_jsonable(vars(args))
    (out / 'sample_ids.json').write_text(json.dumps({
        'train_overfit': [r['id'] for r in train_rows] if args.mode == 'overfit' else None,
        'val': [r['id'] for r in val_rows], 'test': [r['id'] for r in test_rows]}, indent=2))

    def save(filename, step):
        torch.save({'network': network.state_dict(), 'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(), 'step': step,
                    'type': 'official_DecoderGAVP_synthetic_edge', 'settings': config_out,
                    'input_norm': ({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in NORM.items()}
                                   if NORM['mu'] is not None else None)}, out / filename)

    print(f'Experiment: {args.mode}; FlyVis frozen={args.freeze_network}; batch={args.batch_size}; '
          f'train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}', flush=True)
    print('Supervision uses known FORWARD flow vectors.', flush=True)
    history_fields = ['step', 'id', 'loss', 'epe_px', 'zero_epe_px', 'improvement_over_zero_pct',
                      'flow_abs_px', 'sign_accuracy', 'pred_std', 'pred_mean_x', 'pred_mean_y',
                      'network_change_l2', 'network_grad_norm', 'activity_penalty']
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
                n, weight = len(group), len(group) / args.batch_size
                stimulus = torch.cat([sample[0] for _, sample in group])
                target = torch.cat([sample[1] for _, sample in group])
                state = initial_state(network, args, n)
                loss, pen, ms = forward(network, decoder, stimulus, target, args, state,
                                        central_index if use_pen else None)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                if pen is not None:   # gradient for nodes_bias only, kept apart from the EPE gradient
                    g, = torch.autograd.grad(pen * weight, network.nodes_bias, retain_graph=True)
                    pen_grad = g if pen_grad is None else pen_grad + g
                    pen_values.append(float(pen.detach()))
                (loss * weight).backward()
                batch_metrics.extend(ms)
                batch_losses.extend(m['epe_px'] for m in ms)   # per-clip EPE = that clip's loss
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
            delta =math.sqrt(sum(float((x.detach() - orig).square().sum())
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
                val = evaluate(network, decoder, validation_cached, args, device)
                val['step'] = step
                (out / f'validation_step_{step:06d}.json').write_text(json.dumps(val, indent=2))
                print('Validation:', json.dumps(val), flush=True)
                if val['epe_px'] < best:
                    best = val['epe_px']
                    save('best.pt', step)
                    (out / 'best_metrics.json').write_text(json.dumps(val, indent=2))
            f.flush()
    save('final.pt', args.steps)
    final_test = evaluate(network, decoder, test_cached, args, device)
    final_test['step'] = args.steps
    (out / 'test_metrics.json').write_text(json.dumps(final_test, indent=2))
    print('Test:', json.dumps(final_test), flush=True)
    print(f'Finished: {out}; best validation EPE {best:.5f}px', flush=True)


if __name__ == '__main__':
    main()
