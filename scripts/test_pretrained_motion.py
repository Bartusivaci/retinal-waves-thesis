"""
Validate FlyVis motion selectivity using an official pretrained model.

Based on the official FlyVis Moving Edge Responses tutorial:
https://turagalab.github.io/flyvis/examples/04_flyvision_moving_edge_responses/

Run:
    python -m scripts.test_pretrained_motion

This script does NOT train or modify any network.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flyvis import results_dir
from flyvis.network import NetworkView
from flyvis.datasets.moving_bar import MovingEdge

from flyvis.analysis.moving_bar_responses import (
    direction_selectivity_index,
    preferred_direction,
    correlation_to_known_tuning_curves,
)


ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    ROOT / "results" / "motion_debug" / "pretrained_control"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Initialize official moving-edge dataset
# --------------------------------------------------

print("Creating official FlyVis MovingEdge dataset...")

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

print("Number of stimuli:", len(dataset))
print("Stimulus parameters:")
print(dataset.arg_df)


# --------------------------------------------------
# 2. Load pretrained FlyVis
# --------------------------------------------------

model_path = Path(results_dir) / "flow/0000/000"

if not model_path.exists():
    raise FileNotFoundError(
        f"Pretrained model not found: {model_path}\n"
        "Run: flyvis download-pretrained"
    )

print("\nLoading pretrained FlyVis...")
print("Model:", model_path)

network_view = NetworkView(model_path)


# --------------------------------------------------
# 3. Simulate moving-edge responses
# --------------------------------------------------

print("\nComputing moving-edge responses...")

responses = network_view.moving_edge_responses(dataset)

print("Response computation completed.")


# --------------------------------------------------
# 4. Calculate official direction-selectivity indices
# --------------------------------------------------

print("\nCalculating direction selectivity...")

dsis = direction_selectivity_index(responses)

preferred = preferred_direction(responses)

correlations = correlation_to_known_tuning_curves(responses)


# --------------------------------------------------
# 5. Inspect T4/T5 populations
# --------------------------------------------------

results = {}

cell_types = [
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
]

for cell_type in cell_types:

    # T4: ON edges
    # T5: OFF edges
    intensity = 1 if cell_type.startswith("T4") else 0

    dsi_values = (
        dsis.custom.where(
            cell_type=cell_type,
            intensity=intensity
        )
    )

    pd_values = (
        preferred.custom.where(
            cell_type=cell_type,
            intensity=intensity
        )
    )

    corr_values = (
        correlations.custom.where(
            cell_type=cell_type,
            intensity=intensity
        )
    )

    dsi = float(np.nanmean(dsi_values.values))

    direction = float(
        np.rad2deg(
            np.nanmean(pd_values.values)
        )
    )

    correlation = float(
        np.nanmean(corr_values.values)
    )

    results[cell_type] = {
        "intensity": "ON" if intensity == 1 else "OFF",
        "dsi": dsi,
        "preferred_direction_degrees": direction,
        "tuning_correlation": correlation,
    }

    print(
        f"{cell_type}: "
        f"DSI={dsi:.4f}, "
        f"Preferred={direction:.1f}°, "
        f"Correlation={correlation:.4f}"
    )


# --------------------------------------------------
# 6. Plot T4c response traces
# --------------------------------------------------

print("\nPlotting T4c response traces...")

try:

    plt.figure(figsize=(10, 5))

    responses["responses"].custom.where(
        cell_type="T4c",
        intensity=1,
        time=">-0.5,<1.0",
    ).custom.plot_traces(
        x="time",
        legend_labels=["angle"],
    )

    plt.title("Pretrained FlyVis: T4c ON-edge responses")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "T4c_response_traces.png",
        dpi=150,
    )

    plt.close()

except Exception as error:

    print("Trace plotting failed:", error)
    plt.close("all")


# --------------------------------------------------
# 7. Plot official angular tuning
# --------------------------------------------------

print("\nPlotting T4c angular tuning...")

try:

    from flyvis.analysis.moving_bar_responses import (
        plot_angular_tuning
    )

    fig, ax = plot_angular_tuning(
        responses,
        cell_type="T4c",
        intensity=1,
    )

    fig.savefig(
        OUTPUT_DIR / "T4c_angular_tuning.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

except Exception as error:

    print("Angular tuning plotting failed:", error)
    plt.close("all")


# --------------------------------------------------
# 8. Save numerical results
# --------------------------------------------------

summary = {
    "experiment": "pretrained_motion_positive_control",
    "model": str(model_path),
    "stimulus": "official FlyVis MovingEdge",
    "results": results,
}

summary_path = OUTPUT_DIR / "summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2)
)

print("\n======================================")
print("PRETRAINED MOTION EVALUATION COMPLETED")
print("======================================")

print("Results saved to:", OUTPUT_DIR)
print("No training performed.")