from pathlib import Path
from datetime import datetime, timezone
import copy
import random

import numpy as np
import torch
import flyvis

from flyvis.network import Network


# Configuration
SEED = 42

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "flyvis_initial.pt"

# Prevent accidental overwriting of our baseline
if CHECKPOINT_PATH.exists():
    raise FileExistsError(
        f"Baseline already exists: {CHECKPOINT_PATH}\n"
        "Delete or rename it manually if you intend to recreate it."
    )

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Set random seeds for reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# 2. Initialize an untrained FlyVis network
# --------------------------------------------------

print("Initializing FlyVis from scratch...")

network = Network()

# No pretrained checkpoint is loaded.
# The biological connectome remains unchanged.

network.eval()


# --------------------------------------------------
# 3. Collect model information
# --------------------------------------------------

total_parameters = sum(
    p.numel() for p in network.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in network.parameters()
    if p.requires_grad
)

print(f"Neurons: {network.n_nodes}")
print(f"Connections: {network.n_edges}")
print(f"Total parameters: {total_parameters}")
print(f"Trainable parameters: {trainable_parameters}")


# --------------------------------------------------
# 4. Save the baseline
# --------------------------------------------------

checkpoint = {
    "network": copy.deepcopy(network.state_dict()),

    "metadata": {
        "description": "Initial untrained FlyVis baseline",
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "flyvis_version": getattr(
            flyvis, "__version__", "unknown"
        ),
        "pytorch_version": str(torch.__version__),
        "training_steps": 0,
        "pretrained": False,
        "stimulation": "none",
    },
}

torch.save(checkpoint, CHECKPOINT_PATH)

print(f"\nBaseline saved to: {CHECKPOINT_PATH}")


# --------------------------------------------------
# 5. Verify that it can be restored
# --------------------------------------------------

print("\nVerifying saved checkpoint...")

saved = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

restored_network = Network()
restored_network.load_state_dict(saved["network"], strict=True)

original_state = network.state_dict()
restored_state = restored_network.state_dict()

assert original_state.keys() == restored_state.keys()

for key in original_state:
    assert torch.equal(
        original_state[key].cpu(),
        restored_state[key].cpu(),
    ), f"Parameter mismatch: {key}"

print("Verification successful!")
print("Saved and restored parameters match exactly.")