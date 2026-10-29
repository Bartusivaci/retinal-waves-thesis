from pathlib import Path
import zipfile
import re
import csv
import statistics
from collections import defaultdict

ZIP_PATH = Path("dataset/images.zip")
OUTPUT_CSV = Path("wave_manifest.csv")

pattern = re.compile(r"^(\d+)_(\d+)\.png$")

folders = defaultdict(list)

# Read filenames without extracting images
with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    for filename in archive.namelist():

        path = Path(filename)

        match = pattern.match(path.name)

        if not match:
            continue

        folder = path.parent.parent.name

        first = int(match.group(1))
        second = int(match.group(2))

        folders[folder].append({
            "frame": first,
            "timestep": second,
            "filename": filename,
        })

print(f"Found {len(folders)} folders\n")

manifest = []
all_lengths = []

for folder in sorted(folders, key=lambda x: int(Path(x).name) if Path(x).name.isdigit() else x):

    frames = sorted(
        folders[folder],
        key=lambda x: x["frame"]
    )

    waves = []
    current_wave = []

    for frame in frames:

        if current_wave:

            previous = current_wave[-1]

            frame_gap = frame["frame"] - previous["frame"]
            time_gap = frame["timestep"] - previous["timestep"]

            if frame_gap != 1 or time_gap != 1:
                waves.append(current_wave)
                current_wave = []

        current_wave.append(frame)

    if current_wave:
        waves.append(current_wave)

    lengths = [len(wave) for wave in waves]
    all_lengths.extend(lengths)

    print(
        f"Folder {folder}: "
        f"{len(frames)} frames, "
        f"{len(waves)} waves, "
        f"average length {statistics.mean(lengths):.1f}"
    )

    for wave_id, wave in enumerate(waves):

        manifest.append({
            "folder": folder,
            "wave_id": wave_id,
            "start_filename": wave[0]["filename"],
            "end_filename": wave[-1]["filename"],
            "start_frame": wave[0]["frame"],
            "end_frame": wave[-1]["frame"],
            "start_timestep": wave[0]["timestep"],
            "end_timestep": wave[-1]["timestep"],
            "frame_count": len(wave),
        })

# Save CSV
with OUTPUT_CSV.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "folder",
            "wave_id",
            "start_filename",
            "end_filename",
            "start_frame",
            "end_frame",
            "start_timestep",
            "end_timestep",
            "frame_count",
        ]
    )

    writer.writeheader()
    writer.writerows(manifest)

print("\n=== DATASET SUMMARY ===")
print(f"Total folders: {len(folders)}")
print(f"Total frames: {sum(len(v) for v in folders.values()):,}")
print(f"Total waves: {len(manifest):,}")

print(f"Shortest wave: {min(all_lengths)} frames")
print(f"Longest wave: {max(all_lengths)} frames")
print(f"Average length: {statistics.mean(all_lengths):.2f} frames")
print(f"Median length: {statistics.median(all_lengths):.2f} frames")

print(f"\nManifest saved to: {OUTPUT_CSV}")