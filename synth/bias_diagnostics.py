"""Diagnostics for the lesion-T1 bias experiments (CALIPR_LesionBias_V1.ipynb).

* fully_sampled_control(): the same phantom plane as phantom.simulate() (fine grid, true coil
  maps, background / echo / per-TI phase), but with **every** k-space point of the
  reconstruction matrix sampled, noise-free. With full Cartesian sampling, the true coil maps
  and no regularisation, SENSE has a closed form (A^H A is diagonal), so no solver or
  iteration count enters: x = sum_c S_c^* trunc(m e^{i phi} S_c) / sum_c |S_c|^2.
  It shows what the rest of the pipeline (phases, PSIR combination, echo average, fit,
  evaluation) does when the reconstruction is perfect.
* mixing_fit(): is a lesion's reconstructed TI curve a mix of its ideal curve and the
  surrounding WM's ideal curve (the partial-volume signature), and with what weight?
* blur_ideal(): the resolution-limited ideal signal blurred by a Gaussian along chosen axes,
  for the "how much extra blur would explain the bias" check.
Nothing here is used by the reconstructions themselves.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d

import pipeline_utils as pu
import phantom as ph


def _trunc_c(img_fine, matrix, kmax=None):
    """Complex k-space truncation of a fine-grid image to the reconstruction matrix
    (phantom.kspace_truncate without taking the real part). kmax: optional per-axis
    |k| limit (in reconstruction-matrix samples) to mimic a smaller encoded extent."""
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img_fine)))
    c = [nf // 2 - n // 2 for nf, n in zip(img_fine.shape, matrix)]
    K = K[c[0]:c[0] + matrix[0], c[1]:c[1] + matrix[1], c[2]:c[2] + matrix[2]]
    if kmax is not None:
        for ax, km in enumerate(kmax):
            if km is None:
                continue
            k = np.arange(matrix[ax]) - matrix[ax] // 2
            sh = [1, 1, 1]; sh[ax] = -1
            K = K * (np.abs(k) <= km).reshape(sh)
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))
    return (x * np.prod(matrix) / np.prod(img_fine.shape)).astype(np.complex64)


def fully_sampled_control(phantom_dir, plane='COR', factor=ph.FACTOR, seed=0, kmax=None,
                          verbose=True):
    """Noise-free, fully sampled, true-map, unregularised reconstruction of one plane of the
    static phantom. Uses the per-TI phase offsets stored in that phantom's scan info, so the
    phases match the undersampled data exactly. Returns dict with
    img (X,Y,Z,1,TI) per echo group [list], sens (X,Y,Z,coil,1) (unit RSS)."""
    phantom_dir = Path(phantom_dir)
    scans = ph.load_scans(phantom_dir)[plane]
    cfg = json.load(open(phantom_dir / 'phantom_config.json'))
    assert cfg.get('motion') is None, 'static phantom only'
    info0 = scans[0]['info']
    matrix, fov, geom = info0['matrix'], info0['fov_mm'], info0['geom']
    X, Y, Z = ph.grid_coords(matrix, fov, geom, factor)
    lab, T1, M0, _ = ph.tissue_maps(X, Y, Z, seed)
    S = ph.coil_maps(X, Y, Z, ph.load_spec()['n_coils'])
    bg = ph.background_phase(X, Y, Z)
    read_ax = [X, Y, Z][ph.AXIS[geom[0]]]
    echo_phase = [np.zeros_like(bg), (0.4 + 0.01 * read_ax).astype(np.float32)]
    # the coil maps on the reconstruction grid used by the "reconstruction": the same
    # block mean the phantom stores as truth['sens'], normalised like the ENLIVE maps
    sens = pu.unit_rss(ph.block_mean_c(S, factor))                     # (X,Y,Z,coil,1)
    imgs = []
    for e in range(2):
        per_ti = []
        for s in scans:
            i = s['info']
            sig = ph.ir_signal(M0, T1, i['TI_s'], i['TR_s']).astype(np.float32)
            img = sig * np.exp(1j * (bg + echo_phase[e] + i['ti_phase_rad']))
            coil = np.stack([_trunc_c((img * S[..., c]).astype(np.complex64), matrix, kmax)
                             for c in range(S.shape[-1])], -1)          # (X,Y,Z,coil)
            num = np.sum(np.conj(sens[..., 0]) * coil, axis=-1)
            den = np.sum(np.abs(sens[..., 0]) ** 2, axis=-1)
            per_ti.append(np.where(den > 0, num / np.maximum(den, 1e-12), 0))
            if verbose:
                print(f'  FS control {plane} echo {e} TI {1000 * i["TI_s"]:.0f} ms', flush=True)
        imgs.append(np.stack(per_ti, -1)[:, :, :, None, :].astype(np.complex64))   # (X,Y,Z,1,TI)
    return dict(img=imgs, sens=sens)


# ---- curve-mixing analysis -----------------------------------------------------------
def lesion_masks(truth, i, ring_vox=(2, 2, 1)):
    """Evaluation mask of lesion i (as pu.evaluate_t1) and a WM ring around it:
    WM voxels (truth label) within ring_vox voxels of the lesion's footprint, lesion-free."""
    fr = truth['lesion_frac_by_id'][i]
    core = fr >= 0.5 * fr.max()
    foot = fr > 0
    g = np.ogrid[tuple(slice(-r, r + 1) for r in ring_vox)]              # ellipsoidal element
    st = sum((gg / max(r, 1)) ** 2 for gg, r in zip(g, ring_vox)) <= 1.0
    ring = binary_dilation(foot, st) & ~foot & (truth['label'] == 5) & (truth['lesion_fraction'] == 0)
    return core, ring


def mixing_fit(S_rec, S_ideal, core, ring, ref=pu.REF_TI_IDX):
    """Mean curves (normalised to the reference TI) of the lesion core and the WM ring.
    Fits  rec_core = a * ideal_core + (1 - a) * ideal_ring  (least squares over TIs).
    a = 1: no extra mixing; a < 1: the lesion curve is pulled towards the surrounding WM.
    Returns a, the relative residual of that fit, and the four normalised curves."""
    def mc(S, m):
        v = S[m].mean(0)
        return v / v[ref]
    rc, ic, iw = mc(S_rec, core), mc(S_ideal, core), mc(S_ideal, ring)
    d = ic - iw
    a = float(np.dot(rc - iw, d) / np.dot(d, d))
    res = rc - (a * ic + (1 - a) * iw)
    return dict(a=a, rel_resid=float(np.linalg.norm(res) / np.linalg.norm(d)),
                rec_core=rc, ideal_core=ic, ideal_ring=iw, rec_ring=mc(S_rec, ring))


def blur_ideal(S_ideal, fwhm_vox):
    """Gaussian blur of the ideal signal (X,Y,Z,TI); fwhm_vox = (fx, fy, fz) in voxels."""
    out = S_ideal.astype(np.float32).copy()
    for ax, f in enumerate(fwhm_vox):
        if f and f > 0:
            out = gaussian_filter1d(out, f / 2.3548, axis=ax, mode='nearest')
    return out
