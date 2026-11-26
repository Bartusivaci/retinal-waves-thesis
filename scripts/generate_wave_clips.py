"""Cut retinal-wave clips from dataset/images.zip into the same format as the synthetic datasets.

Output: <output>/{train,val,test}/*.npz (key 'frames', [T,64,64] uint8) + manifest.csv + config.json + preview.png,
so scripts/train_edges_selfsupervised.py can train on it unchanged. There are no flow labels:
dx/dy columns are empty and config.json has "labels": false.

Pipeline (see CLAUDE.md 6.18 for the measurements behind the defaults):
- Folders with duplicate frame indices or a manifest/zip count mismatch (two runs mixed) are excluded automatically.
- Segments = runs of consecutive simulation timesteps inside a folder (frames are saved only while there is activity,
  so a segment is roughly one wave event). Segments are rebuilt from the zip index, not from wave_manifest.csv.
- A clip takes --frames frames every --gap timesteps (wave fronts move about 0.4 px/frame at 64x64, so gap 2-3
  matches the 0.6-1.4 px/frame the synthetic models were trained on). Windows inside a segment do not overlap.
- Binary 256x256 frames are area-averaged 4x4 to 64x64 (active fraction 0..1), then mapped linearly to
  luminance --lum-low..--lum-high (default 0.2..0.8, the synthetic data's range) and stored as uint8.
- Clips are kept only if the window's mean active fraction >= --min-mean-active and its peak >= --min-peak-active
  (23% of segments are speckle only).
- Train/val/test are split by FOLDER (no simulation appears in two splits), 80/10/10 with --seed.

    python -m src.wave_zip_index        # once, builds dataset/images_index.npz
    python -m scripts.generate_wave_clips --gap 2 --output dataset/retinal_waves_gap2
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.wave_zip_index import INDEX, WaveZip

ROOT = Path(__file__).resolve().parent.parent


def bad_folders(wz, manifest_path):
    key = wz.folder.astype(np.int64) * 100000 + wz.frame
    uniq, counts = np.unique(key, return_counts=True)
    bad = set((uniq[counts > 1] // 100000).tolist())
    if manifest_path.exists():
        man = pd.read_csv(manifest_path).groupby('folder').frame_count.sum()
        idx = pd.Series(wz.folder).value_counts()
        bad |= set(man.index[man.values != idx.reindex(man.index).fillna(0).values].tolist())
    return sorted(int(b) for b in bad)


def segments(wz, folder):
    """(first index row, length) of each run of consecutive timesteps in a folder (rows sorted by frame)."""
    rows = wz.rows(folder)
    ts = wz.timestep[rows]
    breaks = np.flatnonzero(np.diff(ts) != 1) + 1
    starts = np.concatenate([[0], breaks]); ends = np.concatenate([breaks, [len(rows)]])
    return [(int(rows[s]), int(e - s)) for s, e in zip(starts, ends)]


def to_luminance(binary_frames, low, high):
    t = binary_frames.shape[0]
    frac = binary_frames.reshape(t, 64, 4, 64, 4).mean(axis=(2, 4))
    return np.clip(np.round((low + (high - low) * frac) * 255), 0, 255).astype(np.uint8), frac


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, default=ROOT / 'dataset' / 'retinal_waves_gap2')
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--gap', type=int, default=2, help='simulation timesteps between consecutive clip frames')
    p.add_argument('--train', type=int, default=8000, help='number of training clips')
    p.add_argument('--val', type=int, default=200)
    p.add_argument('--test', type=int, default=200)
    p.add_argument('--max-per-segment', type=int, default=4, help='cap on clips from one wave event (diversity)')
    p.add_argument('--min-mean-active', type=float, default=0.02)
    p.add_argument('--min-peak-active', type=float, default=0.05)
    p.add_argument('--lum-low', type=float, default=0.2)
    p.add_argument('--lum-high', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.frames < 2 or args.gap < 1:
        p.error('frames >= 2 and gap >= 1 required')
    if not INDEX.exists():
        p.error(f'{INDEX} missing; run: python -m src.wave_zip_index')
    out = args.output
    if (out / 'manifest.csv').exists():
        if not args.overwrite:
            raise FileExistsError(f'{out} exists; use --overwrite or a new --output')
        shutil.rmtree(out)

    wz = WaveZip()
    bad = bad_folders(wz, ROOT / 'wave_manifest.csv')
    folders = np.array(sorted(set(np.unique(wz.folder).tolist()) - set(bad)))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(folders))
    n_train, n_val = int(.8 * len(folders)), int(.1 * len(folders))
    split_folders = {'train': folders[perm[:n_train]], 'val': folders[perm[n_train:n_train + n_val]],
                     'test': folders[perm[n_train + n_val:]]}
    span = (args.frames - 1) * args.gap + 1
    print(f'Excluded folders: {bad}; clip span {span} timesteps', flush=True)

    rows_out, stats = [], {}
    for split, want in (('train', args.train), ('val', args.val), ('test', args.test)):
        (out / split).mkdir(parents=True, exist_ok=True)
        # candidate windows: (segment start row, offset in segment, segment id); non-overlapping, capped per segment
        cands = []
        for folder in split_folders[split]:
            for seg_id, (start, length) in enumerate(segments(wz, int(folder))):
                n_win = length // span
                if n_win == 0:
                    continue
                offsets = rng.permutation(n_win)[:args.max_per_segment] * span
                cands.extend((start, int(o), int(folder), seg_id) for o in offsets)
        order = rng.permutation(len(cands))
        tried = kept = 0
        for c in order:
            if kept >= want:
                break
            start, offset, folder, seg_id = cands[c]
            tried += 1
            idx = start + offset + np.arange(args.frames) * args.gap
            binary = np.stack([wz.read_row(int(i)) for i in idx]).astype(np.float32)
            if binary.shape[1:] != (256, 256):
                raise RuntimeError(f'Unexpected frame shape {binary.shape} in folder {folder}')
            video, frac = to_luminance(binary, args.lum_low, args.lum_high)
            act = frac.mean(axis=(1, 2))
            if act.mean() < args.min_mean_active or act.max() < args.min_peak_active:
                continue
            clip_id = len(rows_out)
            name = f'{split}/wave_{clip_id:05d}.npz'
            np.savez_compressed(out / name, frames=video)
            rows_out.append(dict(id=clip_id, split=split, filename=name, dx_px_per_frame='', dy_px_per_frame='',
                                 folder=folder, segment=seg_id, start_timestep=int(wz.timestep[start + offset]),
                                 gap=args.gap, active_mean=round(float(act.mean()), 4),
                                 active_max=round(float(act.max()), 4)))
            kept += 1
        stats[split] = dict(folders=int(len(split_folders[split])), candidates=len(cands), tried=tried, kept=kept,
                            acceptance=round(kept / max(tried, 1), 3))
        print(f'{split}: {stats[split]}', flush=True)
        if kept < want:
            print(f'WARNING: only {kept} of {want} {split} clips passed the activity filter', flush=True)

    with (out / 'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    (out / 'config.json').write_text(json.dumps({
        'count': len(rows_out), 'frames': args.frames, 'side': 64, 'seed': args.seed, 'labels': False,
        'source': 'dataset/images.zip (binary 256x256 retinal-wave simulations)', 'gap_timesteps': args.gap,
        'frame_interval_s_assumed': 0.02, 'luminance': [args.lum_low, args.lum_high],
        'min_mean_active': args.min_mean_active, 'min_peak_active': args.min_peak_active,
        'max_per_segment': args.max_per_segment, 'excluded_folders': bad, 'splits': stats,
        'split_by': 'folder (80/10/10)'}, indent=2))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show = [r for r in rows_out if r['split'] == 'train'][:6]
    fig, axes = plt.subplots(len(show), 10, figsize=(15, 1.6 * len(show)))
    for r_i, r in enumerate(show):
        frames = np.load(out / r['filename'])['frames']
        for c, t in enumerate(np.linspace(0, args.frames - 1, 10).astype(int)):
            axes[r_i, c].imshow(frames[t], cmap='gray', vmin=0, vmax=255); axes[r_i, c].axis('off')
            if r_i == 0:
                axes[r_i, c].set_title(f't{t}', fontsize=8)
    fig.suptitle(f'{out.name}: gap {args.gap}', fontsize=10); fig.tight_layout()
    fig.savefig(out / 'preview.png', dpi=80); plt.close(fig)
    print(f'Wrote {len(rows_out)} clips to {out}', flush=True)


if __name__ == '__main__':
    main()
