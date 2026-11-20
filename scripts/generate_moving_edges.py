"""Generate a deterministic MOVING-EDGE + TEXTURE dataset for FlyVis (v2).

Output: <output>/{train,val,test}/*.npz plus manifest.csv, config.json, preview.png.
Each sample is `frames` grayscale 64x64 uint8 frames (default 20).

What changed vs. v1:
  * The weak single-sinusoid texture is replaced by a FULL-FIELD texture made of
    several random plane waves (different orientations, 8-24 px wavelength at 64x64,
    i.e. roughly 4-12 hexes after the trainer upsamples to 256x256). Every column
    now carries the full 2D velocity in every frame, much like Sintel.
  * The edge's luminance range is compressed so the texture is not clipped away.
  * Labels/documentation: the stored flow is FORWARD flow (+dx, +dy), matching the
    trainer. (v1 docstring/config wrongly said "backward".)
  * --no-edge generates texture-only clips (good first sanity test).

Everything moves rigidly: texture and edge are both functions of the object
coordinates (x - shift*theta), so brightness constancy holds exactly.

Run: python -m scripts.generate_moving_edges --count 1000 \
         --output dataset/synthetic_moving_edges_v2
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


def render_sequence(seed: int, frames: int = 20, side: int = 64, edge: bool = True,
                    directions_deg=None, speed_range=(.6, 1.4)):
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0, 2 * np.pi))
    if directions_deg:  # restrict to a few directions (drawn after, so RNG order is unchanged)
        angle = float(np.deg2rad(directions_deg[int(rng.integers(len(directions_deg)))]))
    theta = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)  # screen x,y
    speed = float(rng.uniform(*speed_range))  # pixels/frame at 64x64
    polarity = int(rng.choice([-1, 1]))
    offset = float(rng.uniform(-5, 5))
    sharpness = float(rng.uniform(.7, 1.8))
    # Compressed luminance range so texture survives the clip.
    low = float(rng.uniform(.20, .35))
    high = float(rng.uniform(.65, .80))
    # Full-field texture: sum of random plane waves, unit SD after normalisation.
    n_waves = 6
    wavelength = rng.uniform(8, 24, n_waves)          # px at 64x64
    wave_dir = rng.uniform(0, 2 * np.pi, n_waves)
    kx = 2 * np.pi / wavelength * np.cos(wave_dir)
    ky = 2 * np.pi / wavelength * np.sin(wave_dir)
    phase = rng.uniform(-np.pi, np.pi, n_waves)
    tex_sd = float(rng.uniform(.10, .18))

    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    xx = xx - (side - 1) / 2
    yy = yy - (side - 1) / 2
    result = []
    for t in range(frames):
        shift = (t - (frames - 1) / 2) * speed
        wx = xx - shift * theta[0]       # object coordinates -> rigid motion in +theta
        wy = yy - shift * theta[1]
        if edge:
            signed = wx * theta[0] + wy * theta[1] - offset
            logistic = 1 / (1 + np.exp(-np.clip(polarity * signed / sharpness, -35, 35)))
        else:
            logistic = np.full_like(wx, .5)
        texture = np.zeros_like(wx)
        for i in range(n_waves):
            texture += np.sin(kx[i] * wx + ky[i] * wy + phase[i])
        texture *= tex_sd / np.sqrt(n_waves / 2)   # each sine has variance 1/2
        image = np.clip(low + (high - low) * logistic + texture, 0, 1)
        result.append(np.round(image * 255).astype(np.uint8))
    video = np.stack(result)
    return video, {'angle_rad': angle,
                   'dx_px_per_frame': float(speed * theta[0]),
                   'dy_px_per_frame': float(speed * theta[1]),
                   'speed_px_per_frame': speed, 'polarity': polarity,
                   'offset_px': offset, 'sharpness_px': sharpness,
                   'tex_sd': tex_sd, 'has_edge': int(edge)}


def generate(root: Path, count: int = 1000, frames: int = 20, side: int = 64,
             seed: int = 2026, overwrite: bool = False, edge: bool = True,
             directions_deg=None, speed_range=(.6, 1.4)):
    if count < 10 or frames < 8 or side < 32:
        raise ValueError('count >= 10, frames >= 8, side >= 32 required')
    rng = np.random.default_rng(seed)
    ids = rng.permutation(count)
    ntrain = int(round(count * .8)); nval = int(round(count * .1))
    split_by_id = {int(k): ('train' if i < ntrain else 'val' if i < ntrain + nval else 'test')
                   for i, k in enumerate(ids)}
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.csv'
    if manifest.exists() and not overwrite:
        raise FileExistsError(f'{manifest} already exists; use --overwrite or a new --output')
    fieldnames = ['id', 'split', 'filename', 'seed', 'angle_rad', 'dx_px_per_frame',
                  'dy_px_per_frame', 'speed_px_per_frame', 'polarity', 'offset_px',
                  'sharpness_px', 'tex_sd', 'has_edge']
    rows = []
    for idx in range(count):
        split = split_by_id[idx]
        item_seed = seed * 100000 + idx
        video, meta = render_sequence(item_seed, frames, side, edge, directions_deg, speed_range)
        relative = f'{split}/edge_{idx:05d}.npz'
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, frames=video)
        rows.append({'id': idx, 'split': split, 'filename': relative, 'seed': item_seed, **meta})
        if (idx + 1) % 100 == 0:
            print(f'Generated {idx + 1}/{count}', flush=True)
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    (root / 'config.json').write_text(json.dumps({
        'count': count, 'frames': frames, 'side': side, 'seed': seed,
        'train': ntrain, 'val': nval, 'test': count - ntrain - nval,
        'frame_interval_s_assumed': .02, 'edge': edge,
        'directions_deg': directions_deg, 'speed_range': list(speed_range),
        'texture': 'full-field, 6 random plane waves, 8-24 px wavelength, rigid',
        'ground_truth': 'constant FORWARD flow=(+dx,+dy) pixels/frame at 64x64'}, indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        example = np.load(root / rows[0]['filename'])['frames']
        fig, axs = plt.subplots(1, 4, figsize=(10, 3))
        for ax, t in zip(axs, [0, 6, 13, frames - 1]):
            ax.imshow(example[t], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'Frame {t + 1}'); ax.axis('off')
        fig.tight_layout(); fig.savefig(root / 'preview.png', dpi=125); plt.close(fig)
    except ImportError:
        pass
    print(f'Created {count} videos: {root} (train={ntrain}, val={nval}, test={count - ntrain - nval})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--side', type=int, default=64)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--output', type=Path, default=Path('dataset/synthetic_moving_edges_v2'))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-edge', action='store_true', help='texture-only clips (no edge)')
    p.add_argument('--directions', default='',
                   help='comma-separated motion directions in degrees, e.g. 0,180 (default: any)')
    p.add_argument('--speed-min', type=float, default=.6)
    p.add_argument('--speed-max', type=float, default=1.4)
    a = p.parse_args()
    dirs = [float(x) for x in a.directions.split(',') if x.strip()] or None
    generate(a.output, a.count, a.frames, a.side, a.seed, a.overwrite, edge=not a.no_edge,
             directions_deg=dirs, speed_range=(a.speed_min, a.speed_max))


if __name__ == '__main__':
    main()
