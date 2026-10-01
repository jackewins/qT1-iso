"""Shared building blocks, taken from T1map_3plane_SR_V1.ipynb (step 1) and
T1map_US4_LLR_V11.ipynb (T1 fit), so every reconstruction method is compared with the
same loading, coil maps, map-set combination and T1 fit. Only the reconstruction step
should differ between methods.

Conventions
-----------
* BART dims: k-space (1, read, pe, coil, 1, TI), trajectory (3, read, pe, 1, 1, TI),
  image (X, Y, Z, coil/map...). TIs are stacked in dim 5 so one `pics` call per plane and
  echo shares one data scaling (keeps amplitudes comparable across TI).
* Pass the trajectory and weights to the BART wrapper as keyword arguments
  (`bart(1, cmd, D, S, t=T, p=P)`). The wrapper writes kwargs as `-t <file>` right after
  the command string, so a bare `-t` at the end of `cmd` would swallow them.
* Echo groups: e = 0 ('even') and e = 1 ('odd') are reconstructed separately (different
  phase), exactly as for the real data.
* Coil maps: self-calibrated per plane with ONE set of coils shared across all TIs
  (`ncalib --shared-col-dims 32`) and two ENLIVE map sets (soft-SENSE). Maps are
  (X, Y, Z, coil, maps).
* Map-set combination for PSIR: each map set is phase-referenced to its own image at the
  reference (longest) TI and the sets are summed -> one signed image per TI (real part).
"""
import os
import sys

import numpy as np

BART_PATH = os.environ.get('BART_TOOLBOX_PATH', '/usr/local/bart')
os.environ['BART_TOOLBOX_PATH'] = BART_PATH
sys.path.append(os.path.join(BART_PATH, 'python'))
from bart import bart, cfl  # noqa: E402

AXIS = {'X': 0, 'Y': 1, 'Z': 2}
ECHOES = ('even', 'odd')
REF_TI_IDX = -1

# The reference method currently being tuned (joint locally low rank, FISTA):
LLR_LAMBDA, LLR_ITER, LLR_BLK = 0.006, 80, 4
LLR_CMD = f'pics -e -S -N -R L:7:7:{LLR_LAMBDA} -i {LLR_ITER} -b {LLR_BLK} -U'


def checked(x):
    """The BART wrapper returns None/0-d on failure; fail loudly instead."""
    x = np.squeeze(x) if x is not None else None
    if x is None or x.ndim < 3:
        raise RuntimeError('BART call failed - see the error printed above')
    return x.astype(np.complex64)


def load_echo(stem, e):
    d = cfl.readcfl(f'{stem}_data')[:, :, :, :, :, e]
    t = cfl.readcfl(f'{stem}_traj')[:, :, :, :, :, e]
    return t, d


def to6(a):
    return a.reshape(a.shape + (1,) * (6 - a.ndim))


def stack_tis(scans_plane, e):
    """Stack the TIs of one plane into dim 5. Unequal PE counts are zero-padded and get a
    zero-weight pattern P (pass as p=P); P is None when all TIs have equal counts."""
    trajs, datas = zip(*[load_echo(s['stem'], e) for s in scans_plane])
    tmax = np.max([t.shape for t in trajs], axis=0)
    dmax = np.max([d.shape for d in datas], axis=0)
    padding = any(tuple(t.shape) != tuple(tmax) for t in trajs)
    T, D, P = [], [], []
    for t, d in zip(trajs, datas):
        T.append(np.pad(t, [(0, a - b) for a, b in zip(tmax, t.shape)]))
        D.append(np.pad(d, [(0, a - b) for a, b in zip(dmax, d.shape)]))
        w = np.zeros((1, dmax[1], dmax[2], 1, 1), np.complex64)
        w[0, :d.shape[1], :d.shape[2], 0, 0] = 1.0
        P.append(w)
    T = np.concatenate([to6(x) for x in T], axis=5)
    D = np.concatenate([to6(x) for x in D], axis=5)
    P = np.concatenate([to6(x) for x in P], axis=5) if padding else None
    return T, D, P


def unit_rss(s):
    """Normalise maps (X,Y,Z,coil[,maps]) to unit root-sum-of-squares over coils and maps."""
    s = s if s.ndim == 5 else s[..., None]
    rss = np.sqrt(np.sum(np.abs(s) ** 2, axis=(3, 4), keepdims=True))
    rss = np.where(rss > 1e-9 * rss.max(), rss, np.inf)
    return (s / rss).astype(np.complex64)


def shared_selfcal(scans_plane, e, nmaps=2):
    """ENLIVE calibration on all TIs of one plane/echo, coils shared across TI (dim 5)."""
    T, D, P = stack_tis(scans_plane, e)
    reso = '{}:{}:{}'.format(*scans_plane[REF_TI_IDX]['info']['matrix'])
    kw = {'t': T} if P is None else {'t': T, 'p': P}
    s = bart(1, f'ncalib -m {nmaps} -x {reso} --shared-col-dims 32', D, **kw)
    s = np.squeeze(s) if s is not None else None
    if s is None or s.ndim < 4:
        raise RuntimeError('ncalib failed - see the error printed above')
    return unit_rss(s)


def combine_maps(x, ref=REF_TI_IDX):
    """(X,Y,Z,M,TI) -> (X,Y,Z,TI): each map set phase-referenced to its reference-TI image."""
    r = x[..., ref]
    num = np.sum(x * np.conj(r)[..., None], axis=3)
    den = np.sqrt(np.sum(np.abs(r) ** 2, axis=3))[..., None]
    return (num / np.maximum(den, 1e-12 * den.max())).astype(np.complex64)


def rss_maps(x):
    """(X,Y,Z,M,TI) -> (X,Y,Z,TI) magnitude, root-sum-of-squares over map sets."""
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=3)).astype(np.float32)


def llr_recon(scans_plane, sens_plane_echo, e, cmd=LLR_CMD):
    """Reference method: joint LLR of all TIs of one plane/echo -> (X,Y,Z,M,TI)."""
    T, D, P = stack_tis(scans_plane, e)
    kw = {'t': T} if P is None else {'t': T, 'p': P}
    x = checked(bart(1, cmd, D, sens_plane_echo, **kw))
    return x if x.ndim == 5 else x[..., None, :]


def ir_signal(T1_s, TI_s, TR_s):
    return 1.0 - 2.0 * np.exp(-TI_s / T1_s) + np.exp(-TR_s / T1_s)


def fit_t1_grid(S, TI_s, TR_s, mask, t1_range_s=(0.02, 5.0), n_grid=2000):
    """Closed-form-M0 dense grid search (V11). S: (..., nTI) signed real signal.
    Returns T1 (s), M0, R^2 with NaN outside `mask`."""
    shape = S.shape[:-1]
    Sf = S.reshape(-1, S.shape[-1]).astype(np.float32)
    grid = np.logspace(*np.log10(t1_range_s), n_grid).astype(np.float32)
    F = np.stack([ir_signal(t, TI_s, TR_s) for t in grid]).astype(np.float32)
    den = np.sum(F ** 2, axis=1)
    T1 = np.full(Sf.shape[0], np.nan, np.float32); M0 = T1.copy(); R2 = T1.copy()
    idx = np.flatnonzero(mask.ravel())
    for chunk in np.array_split(idx, max(1, len(idx) // 8000)):
        s = Sf[chunk]; num = s @ F.T
        rss = (s ** 2).sum(1, keepdims=True) - num ** 2 / den
        k = np.argmin(rss, axis=1); v = np.arange(len(chunk))
        T1[chunk] = grid[k]; M0[chunk] = num[v, k] / den[k]
        sst = ((s - s.mean(1, keepdims=True)) ** 2).sum(1)
        R2[chunk] = 1.0 - rss[v, k] / np.maximum(sst, 1e-12)
    return T1.reshape(shape), M0.reshape(shape), R2.reshape(shape)


def spacing_of(info):
    return tuple(f / n for f, n in zip(info['fov_mm'], info['matrix']))


TRUE_LESION_T1 = {6: 357.5, 7: 192.5}   # label -> T1 (ms); evaluate_t1 reads phantom.TISSUE


def evaluate_t1(T1_s, truth):
    """Compare a T1 map (seconds, on a plane's reconstruction grid) with truth_<PLANE>.npz.

    Tissue masks are the truth labels eroded by one voxel (in-plane and through-plane) so
    partial volume at boundaries does not dominate. Each lesion is evaluated over the
    voxels holding >= half its peak partial-volume fraction. Returns a list of row dicts.
    """
    from scipy.ndimage import binary_erosion
    T1 = 1000.0 * np.asarray(T1_s)
    lab = truth['label']
    rows = []
    for l, name in ((5, 'WM'), (4, 'GM'), (3, 'CSF')):
        m = binary_erosion(lab == l) & np.isfinite(T1)
        v = T1[m]; t = truth['T1_ms'][m]
        rows.append(_row(name, m, T1, truth, float(t.mean())))
    try:
        from phantom import LESIONS, TISSUE
        lesion_t1 = {l: float(TISSUE[l][1][0]) for l in TRUE_LESION_T1}
    except ImportError:
        LESIONS, lesion_t1 = None, TRUE_LESION_T1
    for i, fr in enumerate(truth['lesion_frac_by_id']):
        # each lesion: voxels holding at least half of its peak partial-volume fraction.
        # peak_pv < 1 means no voxel is fully lesion (lesion smaller than a voxel), which
        # caps the contrast any reconstruction can recover.
        m = (fr >= 0.5 * fr.max()) & np.isfinite(T1) if fr.max() > 0 else np.zeros_like(fr, bool)
        d = LESIONS[i][2] if LESIONS else None
        t_true = float(lesion_t1.get(LESIONS[i][0], np.nan)) if LESIONS else np.nan
        r = _row(f'lesion {i + 1} ({d} mm)' if d else f'lesion {i + 1}', m, T1, truth, t_true)
        r['peak_pv'] = float(fr.max())
        rows.append(r)
    return rows


def _row(name, m, T1, truth, t_true):
    """est vs truth, and vs the resolution-limited reference (T1_ideal_ms) if present."""
    est = float(np.nanmean(T1[m])) if m.any() else np.nan
    r = dict(region=name, n=int(m.sum()), true_ms=t_true, est_ms=est,
             sd_ms=float(np.nanstd(T1[m])) if m.any() else np.nan,
             bias_pct=float(100 * (est - t_true) / t_true) if m.any() else np.nan)
    if 'T1_ideal_ms' in truth and m.any():
        ideal = float(np.nanmean(truth['T1_ideal_ms'][m]))
        r.update(ideal_ms=ideal, bias_vs_ideal_pct=float(100 * (est - ideal) / ideal))
    return r


def print_eval(rows, title=''):
    print(title)
    print('true = phantom T1; ideal = fit to fully sampled, noise-free, resolution-limited '
          'signal (partial volume only); bias vs ideal = reconstruction alone')
    print(f"{'region':18s} {'n':>5s} {'true':>7s} {'ideal':>7s} {'est':>7s} {'sd':>6s} "
          f"{'bias%':>6s} {'vs ideal%':>9s} {'peak PV':>7s}")
    for r in rows:
        pv = f"{r['peak_pv']:7.2f}" if 'peak_pv' in r else ''
        print(f"{r['region']:18s} {r['n']:5d} {r['true_ms']:7.1f} {r.get('ideal_ms', np.nan):7.1f} "
              f"{r['est_ms']:7.1f} {r['sd_ms']:6.1f} {r['bias_pct']:6.1f} "
              f"{r.get('bias_vs_ideal_pct', np.nan):9.1f} {pv}")
