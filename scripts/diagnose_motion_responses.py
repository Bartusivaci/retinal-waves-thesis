"""Inspect raw signed neural responses behind official FlyVis DSI scores.

Run: python -m scripts.diagnose_motion_responses
Requires scripts/evaluate_training_progress.py from prior experiment.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis.moving_bar_responses import direction_selectivity_index
from scripts.evaluate_training_progress import ROOT, make_view

OUT = ROOT / 'results/motion_debug/response_diagnostics'
GAP = ROOT / 'results/multi_window/gap5_10000'
CONDITIONS = [
    ('baseline', ROOT/'checkpoints/flyvis_initial.pt'),
    ('gap5_2500', GAP/'checkpoint_step_002500.pt'),
    ('gap5_7500', GAP/'checkpoint_step_007500.pt'),
    ('gap5_9750', GAP/'best.pt'),
    ('gap5_10000', GAP/'final.pt'),
]
POPULATIONS = {'T4b': 1, 'T4c': 1, 'T5b': 0}
ANGLES = (0, 180)


def summarize_raw(values):
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError('Non-finite response values encountered')
    return dict(mean=float(x.mean()), minimum=float(x.min()), maximum=float(x.max()),
                std=float(x.std()), positive_fraction=float(np.mean(x > 0)),
                negative_fraction=float(np.mean(x < 0)),
                positive_peak=float(np.maximum(x, 0).max()),
                signed_peak=float(x.max()), n=int(x.size))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = MovingEdge(offsets=[-10, 11], intensities=[0, 1], speeds=[19],
                         height=80, post_pad_mode='continue', t_pre=1.0,
                         t_post=1.0, dt=1/200,
                         angles=list(np.arange(0, 360, 30)))
    rows = []
    traces = {}
    for label, path in CONDITIONS:
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f'Analyzing {label}', flush=True)
        view, digest = make_view(label, path, OUT)
        response_ds = view.moving_edge_responses(dataset, batch_size=1)
        dsis = direction_selectivity_index(response_ds)
        data = response_ds['responses']
        traces[label] = {}
        for cell, polarity in POPULATIONS.items():
            sel = data.custom.where(cell_type=cell, intensity=polarity)
            x = np.asarray(sel.values)
            summary = summarize_raw(x)
            summary.update(model=label, population=cell, polarity='ON' if polarity else 'OFF',
                           checkpoint_sha=digest,
                           official_dsi=float(np.nanmean(dsis.custom.where(cell_type=cell, intensity=polarity).values)))
            rows.append(summary)
            print(f"  {cell}: DSI={summary['official_dsi']:.7g}; "
                  f"positive={summary['positive_fraction']:.3%}; "
                  f"min={summary['minimum']:.6g}; max={summary['maximum']:.6g}", flush=True)
            traces[label][cell] = {}
            for angle in ANGLES:
                selected = data.custom.where(cell_type=cell, intensity=polarity, angle=angle)
                # Average all axes other than frame, retaining the signed time course.
                dims_to_reduce = [d for d in selected.dims if d != 'frame']
                trace = selected.mean(dim=dims_to_reduce).values
                traces[label][cell][angle] = np.asarray(trace, dtype=float)

    for cell in POPULATIONS:
        fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(11, 2.4*len(CONDITIONS)), sharex=True)
        for ax, (label, _) in zip(axes, CONDITIONS):
            for angle in ANGLES:
                y = traces[label][cell][angle]
                ax.plot(np.arange(len(y)), y, label=f'{angle}°', linewidth=1)
            ax.axhline(0, color='gray', linewidth=0.8)
            ax.set_ylabel(label + '\nresponse')
            ax.legend(loc='upper right')
        axes[-1].set_xlabel('Frame (official MovingEdge response timeline)')
        fig.suptitle(f'{cell}: signed response time courses, opposite directions')
        fig.tight_layout()
        fig.savefig(OUT/f'{cell}_signed_traces.png', dpi=160)
        plt.close(fig)

    (OUT/'response_diagnostics.json').write_text(json.dumps(rows, indent=2))
    with (OUT/'response_diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('Wrote:', OUT, flush=True)

if __name__ == '__main__':
    main()
