"""
Debug motion stimulus generation and FlyVis BoxEye rendering.

Run from repository root:
    python -m scripts.debug_motion_stimulus

This script:
1. Generates ON/OFF moving bars in 8 directions.
2. Renders them through FlyVis BoxEye.
3. Measures temporal changes in photoreceptor inputs.
4. Compares stimuli across movement directions.
5. Saves plots and numerical results.

No FlyVis network is loaded or trained.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from flyvis.datasets.rendering import BoxEye

# Reuse the EXACT stimulus generator from our previous experiment.
from scripts.evaluate_motion_tuning import moving_bar_movie


ROOT = Path(__file__).resolve().parent.parent

ANGLES = tuple(range(0, 360, 45))


# --------------------------------------------------
# 1. Render images into FlyVis photoreceptor inputs
# --------------------------------------------------

def render(eye, movie, device):

    # Convert grayscale images to float32 in [0, 1].
    x = torch.from_numpy(
        movie.astype(np.float32) / 255.0
    )[None].to(device)

    with torch.no_grad():
        y = eye(x)

    expected_shape = (1, len(movie), 1, 721)

    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected BoxEye output: {tuple(y.shape)}. "
            f"Expected: {expected_shape}"
        )

    # Return [time, 721].
    return y[0, :, 0].detach().cpu().numpy()


# --------------------------------------------------
# 2. Main experiment
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="motion_stimulus_debug"
    )

    parser.add_argument(
        "--lead",
        type=int,
        default=10
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=24
    )

    parser.add_argument(
        "--contrast",
        type=int,
        default=75
    )

    parser.add_argument(
        "--travel",
        type=float,
        default=105
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)

    output_dir = (
        ROOT
        / "results"
        / "motion_debug"
        / args.run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Using device:", device)
    print("Output directory:", output_dir)

    # Initialize the same renderer used previously.
    eye = BoxEye(
        extent=15,
        kernel_size=13
    )

    if hasattr(eye, "to"):
        eye = eye.to(device)

    rendered_by_condition = {}
    records = []

    # --------------------------------------------------
    # 3. Generate and analyze all motion directions
    # --------------------------------------------------

    for polarity in ("ON", "OFF"):

        for angle in ANGLES:

            print(
                f"\nProcessing {polarity} {angle} degrees..."
            )

            movie = moving_bar_movie(
                angle,
                polarity,
                lead=args.lead,
                steps=args.steps,
                contrast=args.contrast,
                travel=args.travel
            )

            rendered = render(
                eye,
                movie,
                device
            )

            rendered_by_condition[
                (polarity, angle)
            ] = rendered

            # Exclude the initial neutral frames.
            motion = rendered[args.lead:]

            differences = np.abs(
                np.diff(motion, axis=0)
            )

            # Mean photoreceptor input before movement.
            baseline = rendered[
                :args.lead
            ].mean(axis=0)

            dynamic_rms = float(
                np.sqrt(
                    np.mean(
                        (motion - baseline) ** 2
                    )
                )
            )

            # Compare original Cartesian images.
            original_differences = np.abs(
                np.diff(
                    movie[args.lead:].astype(np.float32),
                    axis=0
                )
            )

            original_mean_change = float(
                original_differences.mean() / 255.0
            )

            # Compare rendered photoreceptor signals.
            rendered_mean_change = float(
                differences.mean()
            )

            rendered_max_change = float(
                differences.max()
            )

            temporal_std = float(
                motion.std(axis=0).mean()
            )

            # Count photoreceptor columns that change
            # appreciably throughout the movie.
            active_columns = int(
                np.sum(
                    np.ptp(
                        motion,
                        axis=0
                    ) > 1e-3
                )
            )

            row = {
                "polarity": polarity,
                "angle_deg": angle,
                "original_consecutive_mean_abs": original_mean_change,
                "rendered_consecutive_mean_abs": rendered_mean_change,
                "rendered_consecutive_max_abs": rendered_max_change,
                "rendered_dynamic_rms": dynamic_rms,
                "rendered_temporal_std_mean": temporal_std,
                "active_columns_over_time": active_columns,
            }

            records.append(row)

            print(row)

            # --------------------------------------------------
            # 4. Generate diagnostic visualizations
            # --------------------------------------------------

            fig, axes = plt.subplots(
                2,
                3,
                figsize=(12, 6)
            )

            indices = [
                args.lead,
                args.lead + args.steps // 2,
                args.lead + args.steps - 1
            ]

            # Top row: Original moving-bar images.
            for j, t in enumerate(indices):

                axes[0, j].imshow(
                    movie[t],
                    cmap="gray",
                    vmin=0,
                    vmax=255
                )

                axes[0, j].set_title(
                    f"Cartesian frame {t}"
                )

                axes[0, j].axis("off")

            # Bottom left:
            # Rendered photoreceptor inputs over time.
            #
            # Column IDs are not guaranteed to be
            # spatially sorted, so this is a time-series
            # image, NOT a spatial hexagonal map.

            axes[1, 0].imshow(
                motion.T,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest"
            )

            axes[1, 0].set(
                xlabel="Motion frame",
                ylabel="Photoreceptor column ID",
                title="Rendered signals (721 columns)"
            )

            # Bottom middle:
            # Difference from the neutral baseline.
            axes[1, 1].plot(
                np.arange(len(motion)),
                np.mean(
                    np.abs(motion - baseline),
                    axis=1
                )
            )

            axes[1, 1].set(
                xlabel="Motion frame",
                ylabel="Mean |signal - neutral|",
                title="Rendered visual drive"
            )

            # Bottom right:
            # Actual changes between consecutive frames.
            axes[1, 2].plot(
                np.arange(1, len(motion)),
                differences.mean(axis=1)
            )

            axes[1, 2].set(
                xlabel="Motion frame",
                ylabel="Mean |frame-to-frame change|",
                title="Motion signal"
            )

            fig.suptitle(
                f"{polarity} {angle}°"
            )

            fig.tight_layout()

            fig.savefig(
                output_dir / f"{polarity}_{angle:03d}.png",
                dpi=120
            )

            plt.close(fig)

    # --------------------------------------------------
    # 5. Compare different movement directions
    # --------------------------------------------------

    distance = {}

    for polarity in ("ON", "OFF"):

        matrix = np.zeros(
            (8, 8),
            dtype=np.float64
        )

        for i, angle_a in enumerate(ANGLES):

            for j, angle_b in enumerate(ANGLES):

                x = rendered_by_condition[
                    (polarity, angle_a)
                ][args.lead:]

                y = rendered_by_condition[
                    (polarity, angle_b)
                ][args.lead:]

                # RMS difference between rendered movies.
                matrix[i, j] = np.sqrt(
                    np.mean(
                        (x - y) ** 2
                    )
                )

        distance[polarity] = matrix.tolist()

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        image = ax.imshow(
            matrix,
            cmap="magma"
        )

        ax.set(
            xticks=range(8),
            yticks=range(8),
            xticklabels=ANGLES,
            yticklabels=ANGLES,
            xlabel="Direction (degrees)",
            ylabel="Direction (degrees)",
            title=f"{polarity} BoxEye direction separation (RMS)"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="RMS signal difference"
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / f"{polarity}_angle_distance.png",
            dpi=150
        )

        plt.close(fig)

        non_diagonal = matrix[
            ~np.eye(8, dtype=bool)
        ]

        print(
            f"\n{polarity} direction separation:"
        )

        print(
            "Minimum RMS difference:",
            non_diagonal.min()
        )

        print(
            "Maximum RMS difference:",
            non_diagonal.max()
        )

    # --------------------------------------------------
    # 6. Save metrics
    # --------------------------------------------------

    csv_path = output_dir / "metrics.csv"

    with csv_path.open(
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    summary = {
        "settings": vars(args),
        "records": records,
        "angle_distance_rms": distance
    }

    json_path = output_dir / "summary.json"

    json_path.write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    print("\n================================")
    print("MOTION STIMULUS DEBUG COMPLETED")
    print("================================")

    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()