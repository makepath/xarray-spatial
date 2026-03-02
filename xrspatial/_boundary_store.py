"""Memmap-backed boundary strip storage for dask tile sweeps.

Stores top/bottom/left/right boundary strips for a 2D tile grid in
flat numpy memory-mapped files, avoiding O(N_tiles) in-memory nested
lists that can OOM on very large inputs.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np


class BoundaryStore:
    """Disk-backed storage for boundary strips of a tiled 2D grid.

    Each of the four sides (top, bottom, left, right) is stored as a
    single contiguous memmap file.  Strip lookup is O(1) via
    precomputed cumulative offsets, and ``get`` returns a zero-copy
    memmap view.

    Parameters
    ----------
    chunks_y : sequence of int
        Tile heights (one per tile row).
    chunks_x : sequence of int
        Tile widths (one per tile column).
    fill_value : float, optional
        Initial fill value for all strips (default 0.0).
    """

    def __init__(self, chunks_y, chunks_x, fill_value=0.0):
        self._tmpdir = tempfile.mkdtemp(prefix='xrs_bdry_')
        self._closed = False
        self._chunks_y = tuple(chunks_y)
        self._chunks_x = tuple(chunks_x)
        n_ty = len(chunks_y)
        n_tx = len(chunks_x)
        total_h = sum(chunks_y)
        total_w = sum(chunks_x)

        # Cumulative offsets for O(1) strip lookup
        self._cum_x = np.zeros(n_tx + 1, dtype=np.int64)
        np.cumsum(chunks_x, out=self._cum_x[1:])
        self._cum_y = np.zeros(n_ty + 1, dtype=np.int64)
        np.cumsum(chunks_y, out=self._cum_y[1:])

        # top/bottom: strip length = chunks_x[ix], indexed by (iy, ix)
        #   shape (n_ty, total_w) — row iy holds all top/bottom strips
        # left/right: strip length = chunks_y[iy], indexed by (iy, ix)
        #   shape (n_tx, total_h) — row ix holds all left/right strips
        for name, shape in [('top', (n_ty, total_w)),
                            ('bottom', (n_ty, total_w)),
                            ('left', (n_tx, total_h)),
                            ('right', (n_tx, total_h))]:
            path = os.path.join(self._tmpdir, f'{name}.dat')
            mm = np.memmap(path, dtype=np.float64, mode='w+', shape=shape)
            mm[:] = fill_value
            mm.flush()
            setattr(self, f'_{name}', mm)

    def get(self, side, iy, ix):
        """Return a memmap view of the boundary strip for tile (iy, ix)."""
        if side == 'top':
            return self._top[iy, self._cum_x[ix]:self._cum_x[ix + 1]]
        elif side == 'bottom':
            return self._bottom[iy, self._cum_x[ix]:self._cum_x[ix + 1]]
        elif side == 'left':
            return self._left[ix, self._cum_y[iy]:self._cum_y[iy + 1]]
        elif side == 'right':
            return self._right[ix, self._cum_y[iy]:self._cum_y[iy + 1]]
        else:
            raise ValueError(f"Unknown side: {side!r}")

    def set(self, side, iy, ix, data):
        """Write *data* into the boundary strip for tile (iy, ix)."""
        self.get(side, iy, ix)[:] = data

    def close(self):
        """Flush memmaps and remove temporary files."""
        if self._closed:
            return
        self._closed = True
        for name in ('top', 'bottom', 'left', 'right'):
            mm = getattr(self, f'_{name}', None)
            if mm is not None:
                del mm
                setattr(self, f'_{name}', None)
        try:
            shutil.rmtree(self._tmpdir)
        except OSError:
            pass

    def __del__(self):
        self.close()

    def snapshot(self):
        """Return a lightweight in-memory copy and close this store.

        The returned ``BoundarySnapshot`` has the same ``.get()``
        interface but holds plain numpy arrays instead of memmaps,
        so no temp files remain referenced.
        """
        snap = BoundarySnapshot(self)
        self.close()
        return snap

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BoundarySnapshot:
    """Read-only in-memory copy of converged boundary strips.

    Created via ``BoundaryStore.snapshot()``; exposes the same
    ``.get(side, iy, ix)`` interface so callers need no changes.
    """

    def __init__(self, store):
        self._cum_x = store._cum_x
        self._cum_y = store._cum_y
        # Copy memmap data to plain numpy arrays
        for name in ('top', 'bottom', 'left', 'right'):
            src = getattr(store, f'_{name}')
            setattr(self, f'_{name}', np.array(src))

    def get(self, side, iy, ix):
        """Return a numpy view of the boundary strip for tile (iy, ix)."""
        if side == 'top':
            return self._top[iy, self._cum_x[ix]:self._cum_x[ix + 1]]
        elif side == 'bottom':
            return self._bottom[iy, self._cum_x[ix]:self._cum_x[ix + 1]]
        elif side == 'left':
            return self._left[ix, self._cum_y[iy]:self._cum_y[iy + 1]]
        elif side == 'right':
            return self._right[ix, self._cum_y[iy]:self._cum_y[iy + 1]]
        else:
            raise ValueError(f"Unknown side: {side!r}")
