"""Run from repository root: python scripts/test_dataset.py"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.retinal_wave_dataset import RetinalWaveDataset

ZIP = Path('dataset/images.zip')
MANIFEST = Path('wave_manifest.csv')

for split in ('train', 'val'):
    ds = RetinalWaveDataset(ZIP, MANIFEST, window_size=20, split=split)
    print(f'{split}: {len(ds):,} 20-frame windows across {len(ds.segments):,} usable segments')
    if len(ds) == 0:
        continue
    sample = ds[0]
    x = sample['frames']
    print(' First sample:', {k: v for k, v in sample.items() if k != 'frames'})
    print(' Shape:', tuple(x.shape), '| Min:', x.min().item(), '| Max:', x.max().item())
    assert x.ndim == 4 and x.shape[:2] == (20, 1)
    assert torch.isfinite(x).all()
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(' Batch shape:', tuple(batch['frames'].shape))
    ds.close()
print('Dataset smoke test passed.')
