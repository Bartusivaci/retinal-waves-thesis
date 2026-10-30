"""Run from repository root: python -m scripts.test_neuron_stimulation"""

from pathlib import Path
import sys
import torch

from flyvis.network import Network
from src.retinal_wave_dataset import RetinalWaveDataset
from src.neuron_stimulation import NeuronWaveMapper

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# 2. Load retinal-wave dataset
# --------------------------------------------------

# Use existing ZIP loader; do not extract or mutate images.zip.
dataset = RetinalWaveDataset(
    zip_path=ROOT / "dataset/images.zip",
    manifest_path=ROOT / "wave_manifest.csv",
    window_size=20,
    split="train",
)

frames = dataset[0]["frames"]
print("Wave image tensor:", tuple(frames.shape))

# --------------------------------------------------
# 3. Load the EXACT saved baseline
# --------------------------------------------------

network = Network()

checkpoint = torch.load(
    ROOT / "checkpoints/flyvis_initial.pt",
    map_location="cpu",
    weights_only=True,
)

network.load_state_dict(checkpoint["network"], strict=True)
network.eval()

# --------------------------------------------------
# 4. Map retinal waves to L1 neurons
# --------------------------------------------------

# Keep mapping on CPU to avoid device indexing issues.
mapper = NeuronWaveMapper(
    network,
    target_types=("L1",),
    gain=1.0,
)

stimulus = mapper(frames)

print("Neural stimulus tensor:", tuple(stimulus.shape))
print("External-drive verification:", mapper.verify(stimulus))

# Verify stimulus before moving to GPU.
assert stimulus.shape == (1, 20, network.n_nodes)
assert torch.isfinite(stimulus).all()

print("Mapping smoke test passed. No weights updated.")

# --------------------------------------------------
# 5. Optional forward pass
# --------------------------------------------------

if sys.argv[1:] == ["--forward"]:

    # Move BOTH the network and stimulus to the same device.
    network = network.to(device)
    stimulus = stimulus.to(device)

    print("\nRunning FlyVis forward pass...")
    print("Stimulus device:", stimulus.device)

    with torch.no_grad():
        activity = network(
            stimulus,
            dt=0.02,
        )

    print("Neural activity:", tuple(activity.shape))
    print(
        "Neural activity finite:",
        bool(torch.isfinite(activity).all()),
    )

    print("\nForward pass successful. No weights updated.")