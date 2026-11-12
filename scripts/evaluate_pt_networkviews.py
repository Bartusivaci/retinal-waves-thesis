"""Evaluate the two thesis .pt checkpoints with FlyVis NetworkView's official
MovingEdge response pipeline (from the repository root):

    python -m scripts.evaluate_pt_networkviews

The pretrained directory is used ONLY as a configuration template.
No pretrained weights are loaded into the thesis networks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Memory

from flyvis import results_dir
from flyvis.network import NetworkView
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis.moving_bar_responses import (
    direction_selectivity_index,
    preferred_direction,
    correlation_to_known_tuning_curves,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "motion_debug" / "thesis_networkviews"
TEMPLATE = Path(results_dir) / "flow/0000/000"
MODELS = {
    "baseline": ROOT / "checkpoints/flyvis_initial.pt",
    "trained_1000": ROOT / "results/multi_window/retinal_waves_1000/final.pt",
}
CELL_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def load_thesis_weights(network, checkpoint_path):
    """Custom NetworkView recovery function. Never falls back to pretrained."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "network" not in checkpoint:
        raise KeyError(f"Missing 'network' weights in {checkpoint_path}")
    network.load_state_dict(checkpoint["network"], strict=True)
    network.eval()
    return network


def make_view(label, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not (TEMPLATE / "config.json").exists():
        # Some datamate versions store config under a different filename.
        if not TEMPLATE.is_dir():
            raise FileNotFoundError(
                f"Missing FlyVis configuration template: {TEMPLATE}\n"
                "Run: flyvis download-pretrained"
            )

    # Unique cache key tied to actual checkpoint bytes, not merely its filename.
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12]
    cache_dir = OUTPUT / f"{label}_{digest}" / "__cache__"
    cache_dir.mkdir(parents=True, exist_ok=True)

    view = NetworkView(
        TEMPLATE,
        recover_fn=load_thesis_weights,
        best_checkpoint_fn=lambda _directory, **_kwargs: checkpoint_path,
    )
    # Do not read/write the official pretrained response cache.
    view.memory = Memory(
        location=cache_dir,
        backend="xarray_dataset_h5",
        verbose=0,
    )
    view.name = f"thesis/{label}/{digest}"
    view._clear_cache()

    # Confirm that the adapter loads exactly the intended checkpoint.
    model = view.init_network()
    expected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["network"]
    actual = model.state_dict()
    if set(expected) != set(actual):
        raise AssertionError("State dict keys differ after loading")
    for key in expected:
        if not torch.equal(expected[key].cpu(), actual[key].detach().cpu()):
            raise AssertionError(f"Checkpoint mismatch: {key}")
    print(f"Verified {label}: {checkpoint_path.name} (SHA256 {digest})")
    return view


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = MovingEdge(
        offsets=[-10, 11],
        intensities=[0, 1],
        speeds=[19],
        height=80,
        post_pad_mode="continue",
        t_pre=1.0,
        t_post=1.0,
        dt=1 / 200,
        angles=list(np.arange(0, 360, 30)),
    )
    print("Official MovingEdge stimuli:", len(dataset))
    results = {}

    for label, path in MODELS.items():
        print(f"\n=== {label} ===")
        view = make_view(label, path)
        # This calls FlyVis's official stimulus_response and xarray pipeline.
        responses = view.moving_edge_responses(dataset, batch_size=1)
        dsi = direction_selectivity_index(responses)
        preferred = preferred_direction(responses)
        correlations = correlation_to_known_tuning_curves(responses)

        model_results = {}
        for ct in CELL_TYPES:
            intensity = 1 if ct.startswith("T4") else 0
            kwargs = {"cell_type": ct, "intensity": intensity}
            d = float(np.nanmean(dsi.custom.where(**kwargs).values))
            p = float(np.rad2deg(np.nanmean(preferred.custom.where(**kwargs).values)))
            c = float(np.nanmean(correlations.custom.where(**kwargs).values))
            model_results[ct] = {
                "polarity": "ON" if intensity else "OFF",
                "dsi": d,
                "preferred_deg": p,
                "tuning_correlation": c,
            }
            print(f"{ct}: DSI={d:.4f}, Preferred={p:.1f}°, Corr={c:.4f}")
        results[label] = model_results

    print("\n=== Trained - Baseline DSI ===")
    delta = {}
    for ct in CELL_TYPES:
        diff = results["trained_1000"][ct]["dsi"] - results["baseline"][ct]["dsi"]
        delta[ct] = float(diff)
        print(f"{ct}: {diff:+.4f}")

    output = {
        "protocol": "FlyVis MovingEdge 12 angles x ON/OFF, dt=0.005 s",
        "baseline_path": str(MODELS["baseline"]),
        "trained_path": str(MODELS["trained_1000"]),
        "results": results,
        "trained_minus_baseline_dsi": delta,
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(output, indent=2))
    print("\nSaved:", OUTPUT / "comparison.json")


if __name__ == "__main__":
    main()
