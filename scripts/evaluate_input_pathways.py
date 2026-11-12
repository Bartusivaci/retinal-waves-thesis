"""Compare baseline, L1-trained and photoreceptor-trained FlyVis on official MovingEdge.

Run from repository root:
  python -m scripts.evaluate_input_pathways

Requires scripts/evaluate_training_progress.py from earlier experiments.
No training occurs; checkpoint loading uses the previously verified NetworkView adapter.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis.moving_bar_responses import (
    direction_selectivity_index,
    preferred_direction,
    correlation_to_known_tuning_curves,
)
from scripts.evaluate_training_progress import ROOT, CELL_TYPES, make_view

OUT = ROOT / "results/motion_debug/input_pathway_comparison"
CHECKPOINTS = {
    "baseline": ROOT / "checkpoints/flyvis_initial.pt",
    "L1_10k_best": ROOT / "results/multi_window/gap5_10000/best.pt",
    "L1_10k_final": ROOT / "results/multi_window/gap5_10000/final.pt",
    "photoreceptor_10k_best": ROOT / "results/multi_window/photoreceptor_gap5_10000/best.pt",
    "photoreceptor_10k_final": ROOT / "results/multi_window/photoreceptor_gap5_10000/final.pt",
}
# These are *raw response* summaries, not signed baseline-subtracted evoked responses.
# In particular, nonpositive raw values can force rectified DSI to 0.
def finite_mean(v):
    v = np.asarray(v, dtype=float)
    finite = v[np.isfinite(v)]
    return float(finite.mean()) if finite.size else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\\n" + "\\n".join(missing))
    dataset = MovingEdge(
        offsets=[-10, 11], intensities=[0, 1], speeds=[19],
        height=80, post_pad_mode="continue", t_pre=1.0, t_post=1.0,
        dt=1 / 200, angles=list(np.arange(0, 360, 30)),
    )

    rows, results = [], {}
    for label, path in CHECKPOINTS.items():
        print(f"\\nEvaluating {label}: {path}", flush=True)
        view, digest = make_view(label, path, OUT)
        response_ds = view.moving_edge_responses(dataset, batch_size=1)
        dsis = direction_selectivity_index(response_ds)
        preferred = preferred_direction(response_ds)
        correlations = correlation_to_known_tuning_curves(response_ds)
        resp = response_ds["responses"]
        results[label] = {"checkpoint": str(path), "sha256_prefix": digest, "populations": {}}
        for cell in CELL_TYPES:
            intensity = 1 if cell.startswith("T4") else 0
            sel = {"cell_type": cell, "intensity": intensity}
            raw = np.asarray(resp.custom.where(**sel).values, dtype=float)
            finite = raw[np.isfinite(raw)]
            if not finite.size:
                raise RuntimeError(f"Non-finite responses: {label}/{cell}")
            row = {
                "model": label, "cell_type": cell,
                "polarity": "ON" if intensity else "OFF",
                "dsi": finite_mean(dsis.custom.where(**sel).values),
                "preferred_deg": (
                    None if finite_mean(preferred.custom.where(**sel).values) is None
                    else float(np.rad2deg(finite_mean(preferred.custom.where(**sel).values)))
                ),
                "tuning_correlation": finite_mean(correlations.custom.where(**sel).values),
                "raw_mean": float(finite.mean()),
                "raw_std": float(finite.std()),
                "raw_min": float(finite.min()),
                "raw_max": float(finite.max()),
                "raw_positive_fraction": float(np.mean(finite > 0)),
                "raw_negative_fraction": float(np.mean(finite < 0)),
            }
            rows.append(row)
            results[label]["populations"][cell] = row
            print(f"  {cell}: DSI={row['dsi']:.8f}; "
                  f"raw mean={row['raw_mean']:+.5f}; "
                  f"positive={row['raw_positive_fraction']:.1%}", flush=True)

    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "comparison.json").write_text(json.dumps({
        "protocol": "Official FlyVis MovingEdge, 12 directions, ON/OFF, dt=0.005 s",
        "note": "DSI uses FlyVis official rectified responses. raw_* statistics are not baseline-subtracted.",
        "models": results,
    }, indent=2))

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(CELL_TYPES))
    names = list(CHECKPOINTS)
    width = 0.84 / len(names)
    for k, name in enumerate(names):
        ys = [results[name]["populations"][ct]["dsi"] for ct in CELL_TYPES]
        ax.bar(x + (k - (len(names)-1)/2) * width, ys, width, label=name)
    ax.set_xticks(x, CELL_TYPES)
    ax.set_ylabel("Official direction selectivity index (DSI)")
    ax.set_title("Motion selectivity by retinal-wave stimulation pathway")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dsi_by_pathway.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    for name in names:
        y = [results[name]["populations"][ct]["raw_mean"] for ct in CELL_TYPES]
        ax.plot(CELL_TYPES, y, marker="o", label=name)
    ax.axhline(0, linewidth=0.8, color="gray")
    ax.set_ylabel("Mean raw neural response (not baseline-subtracted)")
    ax.set_title("Activity operating point by retinal-wave stimulation pathway")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "raw_response_means.png", dpi=160)
    plt.close(fig)

    print(f"\\nSaved results to {OUT}")
    print("Upload comparison.json, dsi_by_pathway.png, raw_response_means.png")


if __name__ == "__main__":
    main()
