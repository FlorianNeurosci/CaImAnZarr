#!/usr/bin/env python
"""claude_patched_caiman (seeding): minicnmfe's seed selection for CNMF-E initialisation.

With CNMFParams init 'mixture_seeding' = True, CNMF-E seeds neurons only at pixels chosen as in
minicnmfe (minicnmfe/initialization.py: mixture_seeds), instead of at every local maximum of
corr * PNR that passes min_corr and min_pnr:

  1. corr * PNR is smoothed with a Gaussian of 0.7 gSig (the correlation image carries
     pixel-scale speckle that the PNR image does not),
  2. its local maxima at least gSig apart are found (skimage peak_local_max),
  3. a two-component Gaussian mixture on the log of their heights keeps the higher component.

min_corr and min_pnr then no longer gate seeding. The seeds are chosen once on the whole field of
view (a per-patch fit has too few maxima for two populations and over-seeds), from a strided frame
sample of at most about 5000 frames, as minicnmfe's init_stride does.
"""
import logging

import numpy as np


def mixture_seeds(cn, pnr, gSig, smooth_factor=0.7):
    """(N, 2) seed pixels (row, col): local maxima of the smoothed corr * PNR in the higher
    component of a two-component mixture on their log heights (port of minicnmfe's
    mixture_seeds, including its fall-back for a degenerate fit)."""
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    from sklearn.mixture import GaussianMixture

    prod = gaussian_filter((cn * pnr).astype(np.float64), smooth_factor * float(gSig))
    pk = peak_local_max(prod, min_distance=max(1, int(round(float(gSig)))))
    if len(pk) < 8:
        return pk
    h = prod[pk[:, 0], pk[:, 1]]
    ok = h > 0
    if ok.sum() < 8:
        return pk[ok]
    x = np.log(h[ok]).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0, n_init=5).fit(x)
    hi = int(np.argmax(gm.means_.ravel()))
    sel = pk[np.flatnonzero(ok)[gm.predict(x) == hi]]
    if len(sel) < max(3, int(0.01 * len(pk))):
        # a heavy tail can leave the high component with a handful of peaks: keep the
        # brightest decile instead (minicnmfe's guard)
        n_keep = max(3, int(round(0.10 * len(pk))))
        order = np.argsort(-prod[pk[:, 0], pk[:, 1]])
        return pk[order[:n_keep]]
    return sel


def frame_sample(images, max_frames=5000):
    """A (n, d1, d2) float32 array of every k-th frame, k = max(1, T // max_frames).
    Works on the memmap view and on a zarr movie (ZarrImages)."""
    import caiman.zarr_store
    T = images.shape[0]
    stride = max(1, T // max_frames)
    if isinstance(images, caiman.zarr_store.ZarrImages):
        block = images.Yr[:, ::stride]                       # (d, n)
        return np.ascontiguousarray(np.reshape(block.T, (block.shape[1],) + images.dims, order='F'),
                                    dtype=np.float32), stride
    return np.array(images[::stride], dtype=np.float32), stride


def global_mixture_seeds(images, gSig, center_psf=True, max_frames=5000):
    """Seeds for the whole field of view, from CaImAn's own corr and PNR images
    (summary_images.correlation_pnr, as init_neurons_corr_pnr computes them) on a frame sample."""
    from caiman.summary_images import correlation_pnr
    logger = logging.getLogger("caiman")
    sample, stride = frame_sample(images, max_frames)
    g = gSig[0] if np.iterable(gSig) else gSig
    cn, pnr = correlation_pnr(sample, gSig=g, center_psf=center_psf, swap_dim=False)
    cn = np.nan_to_num(cn, nan=0.0)
    cn[cn < 0] = 0
    pnr = np.nan_to_num(pnr, nan=0.0)
    pnr[pnr < 0] = 0
    seeds = mixture_seeds(cn, pnr, g)
    logger.info(f"mixture seeding: {len(seeds)} seeds from {sample.shape[0]} frames (stride {stride})")
    return seeds
