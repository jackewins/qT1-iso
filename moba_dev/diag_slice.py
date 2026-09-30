"""Single-readout-slice moba diagnostics (V2 stability work).

    python moba_dev/diag_slice.py <PLANE> <x> "<moba opts>" [echo]

One readout position of a plane (readout iFFT of the phase-demodulated, TI-stacked k-space,
exactly as in Recon_MOBA_V1.ipynb section 7) goes through `moba` with `-d 4`, so the
residual entering every Newton step is logged. Reports:

* per Newton step: the residual `moba` prints (in its own normalised, gridded metric),
  relative to the first step;
* the final data residual in the *raw data* metric, ||y - A(c * rho)|| / ||y||, with moba's
  coils and model images put back on the data's scale (s_psf / s_data), next to the
  noise floor sqrt(N) sigma / ||y||;
* T1 = 1/R1* on the slice vs the resolution-limited ideal per tissue (median and mean) and
  the fraction of brain voxels (GM/WM/lesion) where R1* collapsed (T1 > 1 s).

The prepared slice data are cached in <PHANTOM_DIR>/moba/diag/. Everything here is phantom
only.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'synth'))
import pipeline_utils as pu  # noqa: E402
from pipeline_utils import bart  # noqa: E402
from phantom import load_scans  # noqa: E402

PHANTOM_DIR = Path(os.environ.get('QT1_PHANTOM_DIR', '~/work/phantom')).expanduser()
DIAG_DIR = PHANTOM_DIR / 'moba' / 'diag'
PHASE_TAPER = 0.15
_bw = sys.modules[bart.__module__]


def bart_quiet(nargout, cmd, *args, threads=1, **kwargs):
    """BART wrapper file handling, output captured (safe for parallel calls)."""
    prep = _bw.bart_prepare(nargout, cmd, *args, **kwargs)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    r = subprocess.run(prep['shell_cmd'], capture_output=True, text=True, env=env)
    out = _bw.bart_postprocess(nargout, r.returncode, prep['infiles'], prep['infiles_kw'], prep['outfiles'])
    return out, r.stdout + r.stderr, r.returncode


def ti_phase_offsets(T, D, matrix, width=PHASE_TAPER, ref=pu.REF_TI_IDX):
    """Global phase of each TI relative to `ref` (V1 section 5, doubled-angle estimator)."""
    tr = np.real(T)
    w = np.exp(-0.5 * sum((tr[a] / (width * matrix[a])) ** 2 for a in range(3)))[None]
    g = pu.checked(bart(1, 'nufft -a -d {}:{}:{}'.format(*matrix), T, (D * w).astype(np.complex64)))
    prod = np.sum(g * np.conj(g[..., ref])[..., None], axis=3)
    return 0.5 * np.angle(np.sum(prod.astype(np.complex128) ** 2, axis=(0, 1, 2)))


_prep_cache = {}


def prepare(plane, e):
    """Readout-iFFT'd, phase-demodulated k-space of one plane/echo (cached on disk)."""
    key = (plane, e)
    if key in _prep_cache:
        return _prep_cache[key]
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    scans = load_scans(PHANTOM_DIR)[plane]
    matrix = scans[pu.REF_TI_IDX]['info']['matrix']
    f = DIAG_DIR / f'prep_{plane}_e{e}.npz'
    if f.exists():
        z = np.load(f)
        Dx, T2, dphi = z['Dx'], z['T2'], z['dphi']
    else:
        T, D, _ = pu.stack_tis(scans, e)
        dphi = ti_phase_offsets(T, D, matrix)
        D = (D * np.exp(-1j * dphi).reshape(1, 1, 1, 1, 1, -1)).astype(np.complex64)
        Dx = bart(1, 'fft -i -u 2', D).astype(np.complex64)
        T2 = T[:, :1].copy()
        T2[0] = 0
        np.savez(f, Dx=Dx, T2=T2, dphi=dphi)
    TI_f = np.array([s['info']['TI_s'] for s in scans], np.complex64).reshape(1, 1, 1, 1, 1, -1)
    info = dict(matrix=matrix, noise_sd=scans[pu.REF_TI_IDX]['info']['noise_sd'],
                TI=np.real(TI_f).ravel(), TR=np.array([s['info']['TR_s'] for s in scans]))
    _prep_cache[key] = (Dx, T2, TI_f, info)
    return _prep_cache[key]


def model_images(maps, TI):
    Mss, M0p, R1 = maps[..., 0], maps[..., 1], np.real(maps[..., 2])
    return Mss[..., None] - (Mss + M0p)[..., None] * np.exp(-np.asarray(TI) * R1[..., None])


def run(plane, x, opts, e=0, ti_scale=1.0, threads=1, ksp_sens=None):
    """moba on readout slice x. `opts` are moba options (without -d, --img_dims, -t).
    ti_scale k: the TI file holds k * TI (s), so moba's R1* is R1 / k; it is converted back
    to 1/s in the returned maps. k acts as a parameter scaling of R1* (--other pscale is not
    used by the -L model in BART v1.0.00)."""
    Dx, T2, TI_f, info = prepare(plane, e)
    TI_f = (TI_f * ti_scale).astype(np.complex64)
    nx, ny, nz = info['matrix']
    y = np.ascontiguousarray(Dx[:, x:x + 1])
    cmd = f'moba {opts} -d 4 --img_dims 1:{ny}:{nz}'
    tmpf = None
    if ksp_sens is not None:                           # initial coils (moba k-space representation)
        import uuid
        from pipeline_utils import cfl
        tmpf = str(DIAG_DIR / f'kspsens_{uuid.uuid4().hex}')
        cfl.writecfl(tmpf, np.asarray(ksp_sens, np.complex64))       # (1, 2ny, 2nz, coil)
        cmd += f' --other ksp-sens={tmpf}'
    t0 = time.time()
    res, log, err = bart_quiet(2, cmd, y, TI_f, threads=threads, t=T2)
    dt = time.time() - t0
    if tmpf is not None:
        for ext in ('.cfl', '.hdr'):
            Path(tmpf + ext).unlink(missing_ok=True)
    if err or res is None:
        raise RuntimeError(f'moba failed ({err}):\n{log[-3000:]}')
    xm, s = [np.squeeze(a) for a in res]
    crop = lambda a: a[ny - ny // 2: ny - ny // 2 + ny, nz - nz // 2: nz - nz // 2 + nz]
    maps, sens = crop(xm).astype(np.complex64), crop(s).astype(np.complex64)
    maps[..., 2] *= ti_scale                                   # R1* back to 1/s
    steps = np.array([float(v) for v in re.findall(r'Step: \d+, Res: ([0-9.eE+-]+)', log)])
    sd = re.findall(r'Scaling: ([0-9.eE+-]+)', log)
    sp = re.findall(r'Scaling_psf: ([0-9.eE+-]+)', log)
    scale_data = float(sd[-1]) if sd else 1.0
    scale_psf = float(sp[-1]) if sp else 1.0
    out = dict(maps=maps, sens=sens, steps=steps, scale_data=scale_data, scale_psf=scale_psf,
               runtime_s=dt, cmd=cmd, x=x, plane=plane, e=e, log=log, ti_scale=ti_scale)
    out['final_res'], out['noise_floor'] = data_residual(out, y, T2, info)
    return out


def data_residual(r, y, T2, info):
    """||y - A(c rho)|| / ||y|| in the raw data metric. moba's image x coil product equals the
    true image x 2 s_data / s_psf (the factor 2: unitary FFT on moba's 2x oversampled grid in
    2D; checked on a converged slice, where the residual then equals the noise floor)."""
    ny, nz = info['matrix'][1:]
    S = model_images(r['maps'], info['TI']) * (0.5 * r['scale_psf'] / r['scale_data'])  # (ny, nz, TI)
    img = (r['sens'][:, :, :, None] * S[:, :, None, :])[None, :, :, :, None, :].astype(np.complex64)
    Y, log, err = bart_quiet(1, 'nufft', T2, img)
    if err or Y is None:
        raise RuntimeError('nufft failed: ' + log[-1000:])
    Y = np.asarray(Y).reshape(y.shape)
    w = np.abs(y).sum(axis=3, keepdims=True) > 0
    n = w.sum() * y.shape[3]
    yn = np.linalg.norm(y)
    return float(np.linalg.norm((y - Y) * w) / yn), float(np.sqrt(n) * info['noise_sd'] / yn)


# ---- coil initialisation: image-space maps -> moba's k-space coil representation ------------
SOBOLEV_A, SOBOLEV_B = 880.0, 32.0                  # moba defaults (--sobolev_a, --sobolev_b)


def sobolev_weights(n1, n2, a=SOBOLEV_A, b=SOBOLEV_B):
    """moba/NLINV coil weights on the 2x grid, incl. fftmod and fftscale (noir/model.c)."""
    kl = ((np.arange(n1) - n1 // 2) / n1)[:, None] ** 2 + ((np.arange(n2) - n2 // 2) / n2)[None] ** 2
    w = ((1 + a * kl) ** (-b / 2)).astype(np.complex64).reshape(1, n1, n2)
    return (bart(1, 'fftmod 6', w) / np.sqrt(n1 * n2)).reshape(1, n1, n2, 1)


def coils_from_ksp(c_ksp):
    """moba's forward coil transform (verified against moba's own output to 2e-6):
    c_img = fftmod(ifft_noncentred(w * c_ksp)). c_ksp: (1, 2ny, 2nz, coil)."""
    n1, n2 = c_ksp.shape[1:3]
    w = sobolev_weights(n1, n2)
    return np.asarray(bart(1, 'fftmod 6', bart(1, 'fft -i -n 6', (w * c_ksp).astype(np.complex64))))


def ksp_from_coils(maps, trunc=1e-3):
    """Image-space coil maps (ny, nz, coil) on the reconstruction grid -> moba's k-space coil
    representation on its 2x grid (1, 2ny, 2nz, coil). The maps are embedded in the centre of
    the 2x grid (edge-padded outside the FOV), and only the frequencies where the Sobolev weight
    is > trunc x max are kept (the inverse weight is unbounded, so this band-limits the maps to
    what moba's coil model can represent at a moderate penalty)."""
    ny, nz, nc = maps.shape
    n1, n2 = 2 * ny, 2 * nz
    o1, o2 = ny - ny // 2, nz - nz // 2
    big = np.pad(maps, ((o1, n1 - ny - o1), (o2, n2 - nz - o2), (0, 0)), mode='edge')[None]
    w = sobolev_weights(n1, n2)
    F = np.asarray(bart(1, 'fft -n 6', np.asarray(bart(1, 'fftmod -i 6', big.astype(np.complex64)))))
    keep = np.abs(w) > trunc * np.abs(w).max()
    return np.where(keep, F / (n1 * n2 * np.where(keep, w, 1)), 0).astype(np.complex64)


TISSUES = ((5, 'WM'), (4, 'GM'), (3, 'CSF'), (6, 'les+'), (7, 'les-'))


def slice_eval(r, truth=None):
    """T1 on the slice vs the resolution-limited ideal, per tissue."""
    if truth is None:
        truth = dict(np.load(PHANTOM_DIR / f'truth_{r["plane"]}.npz'))
    x = r['x']
    lab = truth['label'][x]
    ideal = truth['T1_ideal_ms'][x]
    R1 = np.real(r['maps'][..., 2])
    T1 = 1000.0 / np.maximum(R1, 1e-3)
    rows = {}
    for l, n in TISSUES:
        m = lab == l
        if m.sum() == 0:
            continue
        rows[n] = dict(n=int(m.sum()), med=float(np.median(T1[m])), ideal=float(np.median(ideal[m])),
                       mean=float(np.mean(T1[m])), iqr=float(np.subtract(*np.percentile(T1[m], [75, 25]))))
    brain = (lab >= 4)
    rows['collapsed_frac'] = float(np.mean(T1[brain] > 1000)) if brain.any() else np.nan
    return rows


def summary(r, rows=None):
    rows = rows if rows is not None else slice_eval(r)
    st = r['steps']
    s = (f"{r['plane']} x={r['x']:3d} e{r['e']}  {r['runtime_s']:5.1f}s  res/steps0: "
         + ' '.join(f'{v / st[0]:.2f}' for v in st)
         + f" | final {r['final_res']:.3f} (noise {r['noise_floor']:.3f})"
         + f" | collapsed {100 * rows['collapsed_frac']:.1f}%")
    t = '  '.join(f"{k} {v['med']:.0f}/{v['ideal']:.0f} (IQR {v['iqr']:.0f})"
                  for k, v in rows.items() if isinstance(v, dict) and k in ('WM', 'GM', 'les+', 'les-'))
    return s + '\n    T1 median/ideal ms: ' + t


import hashlib

KEEP = ('maps', 'sens', 'steps', 'scale_data', 'scale_psf', 'runtime_s', 'cmd', 'x', 'plane', 'e',
        'ti_scale', 'final_res', 'noise_floor')


def job_key(j):
    """Cache key of a job: plane, slice, echo, options, TI scaling and coil-init tag."""
    h = hashlib.sha1(f"{j['opts']}|{j.get('ti_scale', 1.0)}|{j.get('ksp_tag', '')}".encode()).hexdigest()[:12]
    return f"{j['plane']}_x{j['x']}_e{j.get('e', 0)}_{h}"


def save_cached(j, r):
    (DIAG_DIR / 'cache').mkdir(parents=True, exist_ok=True)
    np.savez(DIAG_DIR / 'cache' / f'{job_key(j)}.npz', **{k: r[k] for k in KEEP if k in r},
             opts=j['opts'], ksp_tag=j.get('ksp_tag', ''))


def load_cached(j):
    f = DIAG_DIR / 'cache' / f'{job_key(j)}.npz'
    if not f.exists():
        return None
    z = np.load(f)
    r = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    r['tag'] = j.get('tag', j['opts'])
    return r


def run_many(jobs, n_parallel=None, verbose=True):
    """jobs: list of dicts with keys plane, x, opts, e (optional: tag, ti_scale, ksp_sens with a
    ksp_tag naming it). Runs them in parallel (one thread each), caches every result in
    DIAG_DIR/cache (keyed by plane, slice, echo, options) and returns them in job order."""
    from concurrent.futures import ThreadPoolExecutor
    n_parallel = n_parallel or os.cpu_count()
    for j in jobs:                                   # prepare serially (shared cache)
        prepare(j['plane'], j.get('e', 0))

    def one(j):
        r = load_cached(j)
        if r is None:
            r = run(j['plane'], j['x'], j['opts'], j.get('e', 0), j.get('ti_scale', 1.0), ksp_sens=j.get('ksp_sens'))
            save_cached(j, r)
        r['tag'] = j.get('tag', j['opts'])
        r['rows'] = slice_eval(r)
        if verbose:
            print(f"[{r['tag']}]\n  " + summary(r, r['rows']), flush=True)
        return r

    with ThreadPoolExecutor(n_parallel) as ex:
        return list(ex.map(one, jobs))


if __name__ == '__main__':
    plane, x, opts = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    e = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    r = run(plane, x, opts, e)
    print(r['cmd'])
    print(summary(r))
