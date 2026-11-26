"""Memory-light random access to dataset/images.zip (about 8 million PNGs, 12 GB).

zipfile.ZipFile builds one Python object per entry for the whole central directory, which needs
several GB for this archive and was OOM-killed under WSL (7 GB). This module streams the central
directory once and stores a compact numpy index (folder, frame, timestep, offset, size, method),
then reads single frames by seeking straight to their local header.

    python -m src.wave_zip_index            # builds dataset/images_index.npz (once, about a minute)
    from src.wave_zip_index import WaveZip
    wz = WaveZip(); img = wz.read(folder=0, frame=5)   # uint8 array
"""
from __future__ import annotations

import io
import re
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / 'dataset' / 'images.zip'
INDEX = ROOT / 'dataset' / 'images_index.npz'
NAME = re.compile(rb'(?:^|/)(\d+)/parsed/(\d+)_(\d+)\.png$')


def _central_directory(f):
    f.seek(0, 2)
    size = f.tell()
    tail_len = min(size, 1 << 20)
    f.seek(size - tail_len)
    tail = f.read(tail_len)
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0:
        raise RuntimeError('No end-of-central-directory record')
    n, cd_size, cd_offset = struct.unpack('<HII', tail[eocd + 10:eocd + 20])
    loc = tail.rfind(b'PK\x06\x07', 0, eocd)          # zip64 locator
    if loc >= 0:
        z64_offset, = struct.unpack('<Q', tail[loc + 8:loc + 16])
        f.seek(z64_offset)
        rec = f.read(56)
        if rec[:4] != b'PK\x06\x06':
            raise RuntimeError('Bad zip64 end record')
        n, = struct.unpack('<Q', rec[32:40])
        cd_size, cd_offset = struct.unpack('<QQ', rec[40:56])
    return n, cd_offset, cd_size


def build_index(zip_path=ZIP, out=INDEX, chunk=64 << 20):
    with open(zip_path, 'rb') as f:
        n, cd_offset, cd_size = _central_directory(f)
        table, count = np.empty((n, 6), dtype=np.int64), 0   # preallocated: a list of tuples needs GBs
        f.seek(cd_offset)
        buf, pos, read = b'', 0, 0
        for _ in range(n):
            if len(buf) - pos < 46 + 65535 * 3 and read < cd_size:
                more = f.read(min(chunk, cd_size - read))
                read += len(more)
                buf, pos = buf[pos:] + more, 0
            if buf[pos:pos + 4] != b'PK\x01\x02':
                raise RuntimeError(f'Bad central directory entry at {pos}')
            method, = struct.unpack('<H', buf[pos + 10:pos + 12])
            csize, usize = struct.unpack('<II', buf[pos + 20:pos + 28])
            nlen, xlen, clen = struct.unpack('<HHH', buf[pos + 28:pos + 34])
            offset, = struct.unpack('<I', buf[pos + 42:pos + 46])
            name = buf[pos + 46:pos + 46 + nlen]
            extra = buf[pos + 46 + nlen:pos + 46 + nlen + xlen]
            pos += 46 + nlen + xlen + clen
            if 0xFFFFFFFF in (csize, usize, offset):       # zip64 extra field
                e = 0
                while e + 4 <= len(extra):
                    tag, ln = struct.unpack('<HH', extra[e:e + 4])
                    if tag == 1:
                        vals, k = extra[e + 4:e + 4 + ln], 0
                        if usize == 0xFFFFFFFF:
                            usize, = struct.unpack('<Q', vals[k:k + 8]); k += 8
                        if csize == 0xFFFFFFFF:
                            csize, = struct.unpack('<Q', vals[k:k + 8]); k += 8
                        if offset == 0xFFFFFFFF:
                            offset, = struct.unpack('<Q', vals[k:k + 8]); k += 8
                        break
                    e += 4 + ln
            match = NAME.search(name)
            if match:
                table[count] = (int(match[1]), int(match[2]), int(match[3]), offset, csize, method)
                count += 1
    a = table[:count]
    a = a[np.lexsort((a[:, 1], a[:, 0]))]                  # sort by folder, frame
    np.savez(out, folder=a[:, 0].astype(np.int32), frame=a[:, 1].astype(np.int32),
             timestep=a[:, 2].astype(np.int32), offset=a[:, 3], csize=a[:, 4], method=a[:, 5].astype(np.int8))
    print(f'Indexed {len(a):,} PNGs of {n:,} zip entries -> {out}')


class WaveZip:
    def __init__(self, zip_path=ZIP, index=INDEX):
        d = np.load(index)
        self.folder, self.frame, self.timestep = d['folder'], d['frame'], d['timestep']
        self.offset, self.csize, self.method = d['offset'], d['csize'], d['method']
        self.f = open(zip_path, 'rb')

    def rows(self, folder):
        lo, hi = np.searchsorted(self.folder, [folder, folder + 1])
        return np.arange(lo, hi)

    def read_row(self, i):
        from PIL import Image
        self.f.seek(int(self.offset[i]))
        header = self.f.read(30)
        if header[:4] != b'PK\x03\x04':
            raise RuntimeError('Bad local header')
        nlen, xlen = struct.unpack('<HH', header[26:30])
        self.f.seek(nlen + xlen, 1)
        data = self.f.read(int(self.csize[i]))
        if self.method[i] == 8:
            data = zlib.decompress(data, -15)
        elif self.method[i] != 0:
            raise RuntimeError(f'Unsupported compression {self.method[i]}')
        return np.asarray(Image.open(io.BytesIO(data)))

    def read(self, folder, frame):
        rows = self.rows(folder)
        hit = rows[self.frame[rows] == frame]
        if not len(hit):
            raise KeyError((folder, frame))
        return self.read_row(hit[0])


if __name__ == '__main__':
    build_index()
