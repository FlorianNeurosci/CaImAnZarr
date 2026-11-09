#!/usr/bin/env python
"""Zarr storage for the movie CNMF reads (claude_patched_caiman, patch 1).

CaImAn's CNMF reads the movie from a C-order memory-mapped file: a (pixels, frames) float32
array whose pixel index runs in Fortran order over the frame (`p = y + d1*x`). This module
stores exactly that array in a zarr v3 store instead, chunked so that the two ways CNMF-E reads
it are both cheap:

  * patch workers read a rectangle of pixels over all frames (`ZarrImages.__getitem__`)
  * the full-FOV ring-model fit (`compute_W`) reads blocks of pixel rows, or blocks of frames
    when the background is downsampled (`ssub_B > 1`)

The values written are the values `save_memmap` would write (same `add_to_movie`, same
`border_to_0` rule), so CNMF on the zarr store gives the same result as CNMF on the memmap.

    fname = caiman.save_zarr(['movie.zarr'], base_name='/data/session/Yr')   # -> '/data/session/Yr.zarr'
    Yr, dims, T = caiman.load_zarr(fname)       # like load_memmap; Yr is a lazy (d, T) view
    images = Yr.images()                        # like np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm.fit(images)

Inputs to `save_zarr` can be file names (anything `caiman.load` reads), numpy arrays, zarr
arrays or paths to zarr arrays, each shaped (T, d1, d2). Zarr inputs and arrays are copied in
blocks of frames, so a long recording is never held in memory; a file is loaded one at a time,
as `save_memmap` does.
"""

import logging
import os
from typing import Optional

import numpy as np

import caiman.paths

FORMAT = "caiman-zarr-Yr"
FORMAT_VERSION = 1


def _zarr():
    import zarr  # imported lazily so `import caiman` does not require zarr
    return zarr


def is_zarr_path(fname) -> bool:
    """True if fname names a zarr store (a directory ending in .zarr or holding zarr metadata)."""
    if not isinstance(fname, (str, os.PathLike)):
        return False
    fname = os.fspath(fname)
    if fname.rstrip("/").endswith(".zarr"):
        return True
    return os.path.isdir(fname) and (os.path.exists(os.path.join(fname, "zarr.json"))
                                     or os.path.exists(os.path.join(fname, ".zarray")))


def is_caiman_zarr(fname) -> bool:
    """True if fname is a store written by save_zarr (as opposed to any other zarr data)."""
    if not is_zarr_path(fname):
        return False
    try:
        z = _zarr().open_array(os.fspath(fname), mode="r")
    except Exception:
        return False
    return dict(z.attrs).get("format") == FORMAT


def _default_chunk_frames(d: int) -> int:
    # each write buffer is d * chunk_frames float32; keep it near 1 GB at most
    return int(max(256, min(4096, (1 << 28) // max(d, 1))))


class ZarrYr:
    """Read-only (d, T) view of a CaImAn zarr store: the counterpart of the memmap `Yr`.

    Indexing returns numpy arrays (`Yr[a:b]` is a (b-a, T) float32 array, `Yr[:, t0:t1]` a
    (d, t1-t0) one, integer arrays select rows), so code that slices `Yr` keeps working.
    `np.asarray(Yr)` reads the whole movie into memory.
    """

    def __init__(self, filename: str):
        zarr = _zarr()
        self.filename = os.fspath(filename)
        self._z = zarr.open_array(self.filename, mode="r")
        attrs = dict(self._z.attrs)
        if attrs.get("format") != FORMAT:
            raise ValueError(f"{self.filename} is not a CaImAn zarr movie (attrs: {attrs}); "
                             "write it with caiman.save_zarr")
        self.dims = tuple(int(x) for x in attrs["dims"])
        self.T = int(self._z.shape[1])
        self.shape = tuple(self._z.shape)
        self.dtype = np.dtype(self._z.dtype)
        self.ndim = 2
        self.order = "C"

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"ZarrYr({self.filename!r}, shape={self.shape}, dims={self.dims})"

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(isinstance(k, (list, np.ndarray)) for k in key):
            key = tuple(np.asarray(k) if isinstance(k, list) else k for k in key)
            return np.asarray(self._z.oindex[key])
        return np.asarray(self._z[key])

    def __array__(self, dtype=None, copy=None):
        logging.getLogger("caiman").warning(f"Reading the whole zarr movie {self.filename} into memory")
        out = np.asarray(self._z[:])
        return out if dtype is None else out.astype(dtype, copy=False)

    @property
    def chunks(self):
        return tuple(self._z.chunks)

    def images(self) -> "ZarrImages":
        """(T, *dims) view, the counterpart of np.reshape(Yr.T, [T] + list(dims), order='F')."""
        return ZarrImages(self)

    def row_blocks(self, block_rows: Optional[int] = None):
        """Yield (start, stop) pixel ranges aligned to the store's pixel chunks."""
        c = self.chunks[0]
        step = max(c, ((block_rows or 4096) // c) * c)
        for a in range(0, self.shape[0], step):
            yield a, min(a + step, self.shape[0])

    def time_blocks(self, block_frames: Optional[int] = None, multiple_of: int = 1):
        """Yield (t0, t1) frame ranges aligned to the frame chunks (and to `multiple_of`)."""
        step = block_frames or self.chunks[1]
        step = max(multiple_of, (step // multiple_of) * multiple_of)
        for a in range(0, self.T, step):
            yield a, min(a + step, self.T)


class ZarrImages:
    """(T, d1, d2[, d3]) view of a CaImAn zarr movie, the object `CNMF.fit` takes.

    `images[t, y0:y1, x0:x1]` reads that box and returns a float32 numpy array equal to
    `np.array(np.reshape(Yr.T, [T] + dims, order='F')[...])` on the memmap, in the same memory
    layout (Fortran order: frames vary fastest), so later arithmetic on it is unchanged.
    """

    def __init__(self, Yr: ZarrYr):
        self.Yr = Yr
        self.filename = Yr.filename
        self.dims = Yr.dims
        self.T = Yr.T
        self.shape = (Yr.T,) + Yr.dims
        self.dtype = Yr.dtype
        self.ndim = 1 + len(Yr.dims)

    def __repr__(self):
        return f"ZarrImages({self.filename!r}, shape={self.shape})"

    def __len__(self):
        return self.T

    def sliced_shape(self, key) -> tuple:
        """Shape of images[key] without reading anything (slices only)."""
        key = tuple(key) + (slice(None),) * (self.ndim - len(key))
        return tuple(len(range(n)[k]) for n, k in zip(self.shape, key))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple(key) + (slice(None),) * (self.ndim - len(key))
        if not all(isinstance(k, slice) for k in key[1:]):
            raise IndexError("ZarrImages supports slices along the spatial axes only")
        t_key = key[0]
        axes = [np.arange(n)[k] for n, k in zip(self.dims, key[1:])]
        grids = np.meshgrid(*axes, indexing="ij")
        # Fortran-order flat index of each pixel of the box, enumerated in Fortran order
        flat = np.ravel_multi_index([g.ravel(order="F") for g in grids], self.dims, order="F")
        if isinstance(t_key, (int, np.integer)):
            block = np.asarray(self.Yr._z.oindex[flat, int(t_key)])          # (npix,)
            return np.asfortranarray(block.reshape([len(a) for a in axes], order="F"))
        block = np.asarray(self.Yr._z.oindex[flat, t_key])                    # (npix, nt)
        out = block.T.reshape((block.shape[1],) + tuple(len(a) for a in axes), order="F")
        return np.asfortranarray(out, dtype=np.float32)

    def __array__(self, dtype=None, copy=None):
        Yr = np.asarray(self.Yr)
        out = np.reshape(Yr.T, [self.T] + list(self.dims), order="F")
        return out if dtype is None else out.astype(dtype, copy=False)

    def to_memory(self) -> np.ndarray:
        """The whole movie in memory, with the memory layout the memmap view has
        (a (T, *dims) view of a C-order (d, T) array)."""
        Yr = np.ascontiguousarray(self.Yr[:, :])
        return np.reshape(Yr.T, [self.T] + list(self.dims), order="F")


def load_zarr(filename: str) -> tuple[ZarrYr, tuple, int]:
    """Open a store written by save_zarr. Returns (Yr, dims, T) like load_memmap."""
    Yr = ZarrYr(filename)
    return Yr, Yr.dims, Yr.T


def _open_input(f, var_name_hdf5):
    """One save_zarr input as something sliceable along time: a zarr array (read lazily), or
    a numpy array (a file is loaded whole, as save_memmap does)."""
    zarr = _zarr()
    if isinstance(f, (str, os.PathLike)) and is_zarr_path(f):
        src = zarr.open_array(os.fspath(f), mode="r")
    elif isinstance(f, (str, os.PathLike, list)):
        src = np.asarray(caiman.load(caiman.paths.fn_relocated(f), fr=1, in_memory=True,
                                     var_name_hdf5=var_name_hdf5))
    else:
        src = f  # numpy array, zarr array, or anything with .shape and slicing
    if len(src.shape) != 3:
        raise ValueError(f"save_zarr expects (T, d1, d2) inputs, got shape {src.shape}")
    return src


def save_zarr(filenames: list,
              base_name: str = 'Yr',
              remove_init: int = 0,
              idx_xy: tuple = None,
              var_name_hdf5: str = 'mov',
              add_to_movie: float = 0.0001,
              border_to_0: int = 0,
              chunk_pixels: int = 64,
              chunk_frames: Optional[int] = None,
              compressor=None) -> str:
    """Write movies into one CaImAn zarr store, the zarr counterpart of
    save_memmap(filenames, base_name, order='C', ...).

    Args:
        filenames: list of file names, numpy arrays, zarr arrays or zarr paths, each (T, d1, d2).
            They are concatenated in time.
        base_name: output path; '.zarr' is appended if missing. Without a directory it goes to
            CaImAn's temp dir, as save_memmap does.
        remove_init, idx_xy, var_name_hdf5, add_to_movie, border_to_0: as in save_memmap. As there,
            border_to_0 sets the border of each input to (min of that input) + 1 and add_to_movie is
            added (in float32) after that.
        chunk_pixels, chunk_frames: zarr chunk shape along pixels and frames. 64 pixels keeps patch
            reads tight; chunk_frames defaults to about 1 GB of frames per write buffer.
        compressor: a zarr v3 codec for the chunks; default Blosc lz4 + bitshuffle (lossless).

    Returns:
        the path of the store.
    """
    logger = logging.getLogger("caiman")
    zarr = _zarr()
    if not isinstance(filenames, list):
        raise Exception('save_zarr: input should be a list of filenames or arrays')
    out = os.fspath(base_name)
    if not out.rstrip("/").endswith(".zarr"):
        out = out + ".zarr"
    out = caiman.paths.fn_relocated(out)
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists")
    if compressor is None:
        compressor = zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")
    add = np.float32(add_to_movie)
    z = None
    Ttot = 0
    tmp = out + ".part"
    try:
        for f in filenames:
            src = _open_input(f, var_name_hdf5)
            T_in = int(src.shape[0])
            if idx_xy is None:
                ys, xs = slice(None), slice(None)
            elif len(idx_xy) == 2:
                ys, xs = idx_xy
            else:
                raise NotImplementedError('save_zarr handles 2D movies only')
            frames = range(remove_init, T_in)
            border_val = None
            if border_to_0 > 0:
                # save_memmap: Yr.calc_min() is (nanmin of the movie) + 1, assigned to the border
                mn = np.inf
                for a in range(frames.start, frames.stop, 2000):
                    blk = np.asarray(src[a:min(a + 2000, frames.stop)])[:, ys, xs]
                    mn = min(mn, float(np.nanmin(blk)))
                border_val = mn + 1
            first = np.asarray(src[frames.start:frames.start + 1])[:, ys, xs]
            dims = first.shape[1:]
            d = int(np.prod(dims))
            if z is None:
                cf = chunk_frames or _default_chunk_frames(d)
                cp = int(min(chunk_pixels, d))
                z = zarr.create_array(tmp, shape=(d, 0), chunks=(cp, cf), dtype="float32",
                                      compressors=[compressor], fill_value=0.0,
                                      attributes=dict(format=FORMAT, version=FORMAT_VERSION,
                                                      dims=list(dims), order="C",
                                                      pixel_order="F",
                                                      add_to_movie=float(add_to_movie)))
                dims_ref = dims
            elif dims != dims_ref:
                raise ValueError(f"input dims {dims} differ from the first input's {dims_ref}")
            n_new = len(frames)
            z.resize((d, Ttot + n_new))
            # write in blocks that line up with the frame chunks of the store
            cf = z.chunks[1]
            pos = Ttot
            a = frames.start
            while a < frames.stop:
                b = min(frames.stop, a + (cf - (pos % cf)))
                blk = np.array(np.asarray(src[a:b])[:, ys, xs], dtype=np.float32)
                if border_val is not None:
                    blk[:, :border_to_0, :] = border_val
                    blk[:, :, :border_to_0] = border_val
                    blk[:, :, -border_to_0:] = border_val
                    blk[:, -border_to_0:, :] = border_val
                Y = np.reshape(np.transpose(blk, (1, 2, 0)), (d, b - a), order='F')
                z[:, pos:pos + (b - a)] = np.ascontiguousarray(Y, dtype=np.float32) + add
                pos += b - a
                a = b
            Ttot += n_new
            logger.debug(f"save_zarr: {Ttot} frames written")
        os.rename(tmp, out)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out
