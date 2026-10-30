"""Memory-efficient dataset of uninterrupted windows inside retinal-wave images.zip."""
import bisect
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RetinalWaveDataset(Dataset):
    def __init__(self, zip_path, manifest_path, window_size=20, stride=None,
                 split='train', train_fraction=0.9, split_seed=42):
        if window_size < 2:
            raise ValueError('window_size must be at least 2 for temporal learning')
        self.zip_path = Path(zip_path)
        self.manifest_path = Path(manifest_path)
        self.window_size = int(window_size)
        self.stride = int(stride if stride is not None else window_size)
        if self.stride < 1:
            raise ValueError('stride must be positive')
        if split not in ('train', 'val', 'all'):
            raise ValueError("split must be 'train', 'val', or 'all'")
        if not (0 < train_fraction < 1):
            raise ValueError('train_fraction must be between 0 and 1')
        if not self.zip_path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError('Check the ZIP and manifest paths')

        # Partition complete folders so related frames cannot leak across splits.
        with self.manifest_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        folders = sorted({r['folder'] for r in rows}, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
        generator = np.random.default_rng(split_seed)
        permutation = generator.permutation(len(folders))
        cutoff = int(len(folders) * train_fraction)
        train_set = {folders[i] for i in permutation[:cutoff]}
        selected = train_set if split == 'train' else set(folders) - train_set

        self.segments = []
        self.cumulative = []
        total = 0
        for r in rows:
            if split != 'all' and r['folder'] not in selected:
                continue
            length = int(r['frame_count'])
            start_frame = int(r['start_frame'])
            end_frame = int(r['end_frame'])
            start_time = int(r['start_timestep'])
            end_time = int(r['end_timestep'])
            # The manifest describes contiguous groups. Fail loudly if corrupt.
            if length != end_frame - start_frame + 1 or length != end_time - start_time + 1:
                raise ValueError(f'Non-contiguous manifest row: {r}')
            count = max(0, (length - self.window_size) // self.stride + 1)
            if count == 0:
                continue
            # Zip paths reconstructed from manifest start path, not hard-coded folder names.
            prefix = str(Path(r['start_filename']).parent).replace('\\', '/')
            self.segments.append((r['folder'], int(r['wave_id']), prefix, start_frame, start_time, count))
            total += count
            self.cumulative.append(total)
        self._zip = None  # Open lazily in each DataLoader worker.

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def _archive(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment_index = bisect.bisect_right(self.cumulative, index)
        preceding = self.cumulative[segment_index - 1] if segment_index else 0
        folder, seg_id, prefix, first_frame, first_time, _ = self.segments[segment_index]
        offset = (index - preceding) * self.stride
        first_frame += offset
        first_time += offset
        frames = []
        archive = self._archive()
        for t in range(self.window_size):
            name = f'{prefix}/{first_frame+t}_{first_time+t}.png'
            with archive.open(name) as stream:
                with Image.open(io.BytesIO(stream.read())) as img:
                    arr = np.array(img.convert('L'), dtype=np.uint8, copy=True)
            frames.append(torch.from_numpy(arr).float().div_(255))
        stacked = torch.stack(frames).unsqueeze(1)  # [T, 1, H, W]
        return {'frames': stacked, 'folder': folder, 'segment_id': seg_id,
                'start_frame': first_frame, 'start_timestep': first_time}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zip'] = None
        return state

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __del__(self):
        self.close()
