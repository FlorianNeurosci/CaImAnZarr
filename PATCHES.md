# CaImAnZarr

CaImAn **1.13.2** with four changes for long 1-photon (CNMF-E) recordings and a slimmer
installation. Branch `claude-patched-v1.13.2` is the official `v1.13.2` release plus the commits
below; `main` is an unmodified copy of upstream.

| commit | change | effect on results |
|---|---|---|
| Patch 1 | CNMF reads the movie from a **zarr** store instead of a memmap | bit-identical |
| Patch 2 | **less RAM** in `compute_W` and `local_correlations_fft` | bit-identical |
| Speed 1 | the per-pixel ring regression of the final `compute_W` runs in **forked worker processes** | bit-identical |
| Seeding | optional init parameter `mixture_seeding` (default `False`): seeds chosen as in minicnmfe | off: bit-identical; on: different seeding by design |
| Packaging | `zarr >= 3`; torch, keras, plotting/GUI, NWB and dev tools become extras; their module-level imports are guarded | none (installation only) |

```
git diff v1.13.2..claude-patched-v1.13.2 -- caiman     # the whole change
```

## The changes

**Patch 1: zarr input** (`caiman/zarr_store.py`, new; small edits in `cnmf.py`, `map_reduce.py`,
`movies.py`). `caiman.save_zarr(files, base_name)` writes the same `(pixels, frames)` float32 array
`save_memmap` writes (same `add_to_movie`, same `border_to_0` rule), losslessly compressed (Blosc
lz4 + bitshuffle; about half the size of the `.mmap`). It copies the input in blocks of frames, so a
long recording is never held in RAM. `caiman.load_zarr` returns a lazy `(pixels, frames)` view like
`load_memmap`; patch workers read only their own patch, with the same values and memory layout
(Fortran order) as the copy taken from a memmap.

**Patch 2: less RAM.** At the end of `cnm.fit`, the full-FOV `compute_W` built
`X = Y - A.dot(C) - b0`, holding `A.dot(C)`, `Y - A.dot(C)` and pages of `Y` at once (float64 after
merging). Now `A.dot(C)` is written straight into the buffer that becomes `X`
(`np.dot(A, C, out=X)`, the same BLAS call) and `Y` and `b0` are subtracted in place, block by
block. `A.dot(C)` is deliberately *not* computed in blocks: OpenBLAS then sums in a different order
and the last bit changes. `local_correlations_fft` (run inside every patch worker) fills one
preallocated array instead of stacking a list, and can reuse its input.

**Speed 1: parallel ring fit.** `compute_W(..., parallel=True, n_processes=n)` runs the per-pixel
ring regressions in a `fork` pool (blocks of 512 pixels; the closure reaches the workers by fork, so
the large arrays are shared copy-on-write, not pickled). `cnm.fit` passes its `n_processes`. Where
`fork` is not available (Windows), the original code path runs.

**Seeding: `mixture_seeding`.** With `CNMFParams(params_dict={..., 'mixture_seeding': True})`,
CNMF-E seeds neurons where minicnmfe would instead of at every corr × PNR maximum that passes
`min_corr` / `min_pnr`: local maxima of corr × PNR (smoothed at 0.7 gSig, at least gSig apart),
split by a two-component Gaussian mixture on their log heights, computed once on the whole field of
view from at most about 5000 frames (`caiman/source_extraction/cnmf/seeding.py`). Each patch gets
the seeds inside it; `min_corr` / `min_pnr` then gate nothing, a seed inside an accepted neuron is
skipped, and there is no residual-search round. With seeds, each patch gets its own deep copy of
the parameters (see *A CaImAn issue* below).

## Use

```python
import caiman as cm
from caiman.source_extraction.cnmf import cnmf, params

fname = cm.save_zarr(['movie.tif'], base_name='out/Yr')    # any movies caiman.load reads, arrays or zarr (T, d1, d2)
Yr, dims, T = cm.load_zarr(fname)
opts = params.CNMFParams(params_dict={'fnames': [fname], 'method_init': 'corr_pnr', 'center_psf': True,
                                      'rf': 32, 'stride': 16, 'nb': 0, 'nb_patch': 0, ...})
c, dview, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', n_processes=6)
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
cnm.fit(Yr.images())
```

### Install

```bash
pip install "caiman @ git+https://github.com/FlorianNeurosci/CaImAnZarr@claude-patched-v1.13.2"
```

This builds CaImAn from source (the deconvolution extension needs a C/C++ compiler, e.g. gcc/g++)
and installs only what `import caiman`, motion correction and CNMF / CNMF-E need: about 1.1 GB in a
fresh virtual environment, against 7.6 GB for upstream's dependency list. Extras, as in
`pip install "caiman[dl,viz] @ git+..."`:

| extra | adds | needed for |
|---|---|---|
| `dl` | keras, torch, torchvision | CNN component classifiers, VolPy, OnACID's ring-CNN background |
| `viz` | bokeh, holoviews, moviepy, panel, pyside6 | interactive plots, movie playback and export |
| `nwb` | pynwb | NWB import / export |
| `dev` | pytest, mypy, cython, ... | tests and development |
| `jupyter` | jupyter, ipykernel, ... | notebooks |
| `all` | all of the above | upstream's full set |

Without an extra, its packages are not imported; the functions that need them fail when called.

## Validation

Against unmodified CaImAn 1.13.2 (conda-forge build, Python 3.12, numpy 2.5, OpenBLAS, Linux x86-64,
one BLAS thread per process, 6 worker processes). Every field of `cnm.estimates` is compared element
by element (sparse matrices down to `indptr`/`indices`/`data`) and by SHA-256 of dtype, shape and
bytes. The original run twice on the same input is identical to itself, so exact equality is a fair
test.

| case | frames | components | fork vs original |
|---|---|---|---|
| original run twice (determinism baseline), simulated movie 1 | 2,000 | 28 | identical |
| simulated movie 1 / 2 | 2,000 | 28 / 28 | identical / identical |
| simulated movie 3, `ssub_B` 2 (downsampled background) | 2,000 | 30 | identical |
| simulated movie 1 without patches (`rf=None`), vs the original with the movie in memory | 2,000 | 28 | identical |
| real WT recording, excerpt | 5,000 | 133 | identical |
| real SOM recording, excerpt | 5,000 | 231 | identical |
| real PV recording, excerpt, `ssub_B` 2 | 5,000 | 66 | identical |
| real WT recording, full length (246 × 283 px) | 49,158 | 252 | identical |

On the full-length recording the fork took 28.7 min against 63.1 min for the original (the
original's time is inflated: other jobs shared the machine during its patch phase) and peaked at
29.7 GB of RAM (whole process tree; main process 27.7 GB) against 57.3 GB (51.6 GB).

The same check in a virtual environment with only PyPI packages, everything built with pip from
source: the fork (default slim install and full install) against unmodified 1.13.2 on simulated
movie 1 and the WT excerpt, identical; CaImAn's `MotionCorrect` (rigid, high-pass filtered, two
passes) gives an identical corrected movie and identical shifts with the original, the full and
the slim fork. The hashes equal those from the conda environment.

Unit tests, patched vs original function: `compute_W` 96/96 input combinations (dense/CSC `A`,
float32/64, `ssub_B` 1/2, C/F/zarr `Y`), the parallel ring fit vs the serial one 36/36,
`local_correlations_fft` 32/32, all bit-identical.

Speed and memory, measured on the same code paths on CaImAn 1.13.1 (8-core i7-10700, 6 workers,
1-photon recordings binned to about 250 × 270 px):

| recording | frames | fit, original → patched | peak RAM (process tree), original → patched |
|---|---|---|---|
| WT | 49,158 | 44.5 → 26.8 min | 57.6 → 29.9 GB |
| SOM | 31,213 | 25.9 → 15.6 min | 34.7 → 18.3 GB |
| PV | 41,357 | 19.8 → 12.7 min | 28.2 → 22.8 GB |

The speed-up is the final ring fit (25.2 → 8.3 min on the WT recording); the patch phase is
unchanged. More than 6–8 workers did not help on this 8-core machine.

## A CaImAn issue (also in the original)

The result of `cnm.fit` with patches depends on `n_processes`. `cnmf_patches` makes a shallow
`copy(params)`, so patches that `Pool.map_async` puts in one chunk share the parameter groups, and
each patch's fit writes into them (e.g. `n_pixels_per_process`, computed from the first patch's
pixel count and the free RAM). The default chunk size is `ceil(n_patches / (4 · n_workers))`, so
it changes with the worker count. On a 30-patch recording, the original gave 252 components with
6 workers (chunks of 2 patches) and 253 with 8 (chunks of 1); one patch per task with 6 workers
reproduced the 8-worker result bit for bit. This fork keeps CaImAn's behaviour (so its results
equal the original's at any worker count); with `mixture_seeding=True` each patch gets its own copy.

## Limits

- zarr input covers CNMF-E with patches and `nb_patch=0`, and fits without patches (`rf=None`). The
  other patch branches raise `NotImplementedError`. `evaluate_components` still needs a memmap or an
  in-memory movie; `MotionCorrect` still writes memmaps.
- `mixture_seeding` has been compared with minicnmfe on real recordings, not against ground truth.
  It covers the whole field and gives no duplicates, but it can miss cells in dense regions (no
  residual round).
- Identity was tested on one machine and library stack.

License: GPL-2.0, as CaImAn.
