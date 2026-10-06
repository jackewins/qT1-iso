"""Full-plane, slice-wise `moba -L` reconstruction (V2 design), shared by Recon_MOBA_V2.ipynb
and the tuning runs (Recon_MOBA_Tuning_S1.ipynb, run_stage1.py).

One plane and echo group: stack the 4 TIs (pu.stack_tis), demodulate the per-TI global phase,
inverse FFT along the fully sampled Cartesian readout, then one 2D `moba` problem per readout
slice (4 TIs, trajectory (0, k_pe1, k_pe2)), run in parallel. Scaling: fixed per plane,
`--scale_data K/sigma` (sigma = k-space noise SD) and `--scale_psf` = moba's own PSF
normalisation (Recon_MOBA_V2.ipynb sections 7-8). After each slice the raw-data residual and its
noise floor are stored (convergence QC). Results are cached in <proc_dir> under a name that
holds the options and the phantom fingerprint.
"""
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import diag_slice as ds
import pipeline_utils as pu
from pipeline_utils import bart

NOISE_REF_SCALE = 4.545        # --scale_data = NOISE_REF_SCALE / sigma_k (45 in the coronal plane)
PHASE_TAPER = 0.15


phantom_fingerprint = ds.phantom_fingerprint     # keys every cache by the phantom version


def opts_tag(opts):
    """Readable, filename-safe tag of a moba option string."""
    t = re.sub(r'[^A-Za-z0-9.=:]+', '_', opts.replace('-', '')).strip('_')
    return t if len(t) <= 80 else t[:60] + '_' + hashlib.sha1(opts.encode()).hexdigest()[:8]


def ti_file(scans_plane):
    return np.array([s['info']['TI_s'] for s in scans_plane], np.complex64).reshape(1, 1, 1, 1, 1, -1)


def plane_scaling(Dx, T2, TI_f, matrix, noise_sd, noise_ref_scale=NOISE_REF_SCALE):
    """(s_data, s_psf): s_psf = moba's own PSF normalisation (1000/||PSF||, read from a one-step run
    on the central slice; the PSF is the same for every slice), s_data = noise_ref_scale / sigma."""
    nx, ny, nz = matrix
    _, log, err = ds.bart_quiet(2, f'moba -L -i 1 -C 5 --normalize_scaling --scale_data 5000 --scale_psf 1000 '
                                   f'--img_dims 1:{ny}:{nz}', np.ascontiguousarray(Dx[:, nx // 2:nx // 2 + 1]), TI_f, t=T2)
    s_psf = float(re.findall(r'Scaling_psf: ([0-9.eE+-]+)', log)[-1])
    return noise_ref_scale / noise_sd, s_psf


def moba_slice(y, T2, TI_f, ny, nz, cmd, info_slice, progress=None):
    res, log, err = ds.bart_quiet(2, cmd, y, TI_f, t=T2)
    if err or res is None:
        raise RuntimeError(f'moba failed ({err}):\n{log[-2000:]}')
    x, s = [np.squeeze(a) for a in res]                              # (2ny, 2nz, 3), (2ny, 2nz, coil)
    crop = lambda a: a[ny - ny // 2: ny - ny // 2 + ny, nz - nz // 2: nz - nz // 2 + nz]
    num = lambda key: float(re.findall(rf'{key}: ([0-9.eE+-]+)', log)[-1])
    r = dict(maps=crop(x).astype(np.complex64), sens=crop(s).astype(np.complex64),
             scale_data=num('Scaling'), scale_psf=num('Scaling_psf'),
             steps=np.array([float(v) for v in re.findall(r'Step: \d+, Res: ([0-9.eE+-]+)', log)]))
    r['final_res'], r['noise_floor'] = ds.data_residual(r, y, T2, info_slice)
    if progress is not None:                                         # one line per slice (long runs)
        with open(progress, 'a') as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {info_slice['tag']} res/noise {r['final_res'] / r['noise_floor']:.2f}\n")
    return r


def recon_file(proc_dir, plane, e, opts, phantom_dir, noise_ref_scale=NOISE_REF_SCALE, phase_corr=True):
    tag = f'{opts_tag(opts)}_k{noise_ref_scale}_ph{int(phase_corr)}_p{phantom_fingerprint(phantom_dir)}'
    return Path(proc_dir) / f'moba_{plane}_e{e}_{tag}.npz'


def moba_recon(scans_plane, e, plane, opts, phantom_dir, proc_dir, noise_ref_scale=NOISE_REF_SCALE,
               phase_corr=True, jobs=None, slices=None, progress=None):
    """Slice-wise moba of one plane/echo group. `opts`: moba options without scaling, -d, --img_dims.
    slices: optional subset of readout positions (others are left zero/NaN-free empty).
    Returns (result dict, 'cached' | 'reconstructed')."""
    f = recon_file(proc_dir, plane, e, opts, phantom_dir, noise_ref_scale, phase_corr)
    if slices is not None:
        f = f.with_name(f.stem + '_xs' + hashlib.sha1(','.join(map(str, sorted(slices))).encode()).hexdigest()[:8] + '.npz')
    if f.exists():
        return dict(np.load(f)), 'cached'
    info = scans_plane[pu.REF_TI_IDX]['info']
    matrix = info['matrix']
    nx, ny, nz = matrix
    T, D, P = pu.stack_tis(scans_plane, e)                   # P not passed (moba weights non-zero samples)
    dphi = ds.ti_phase_offsets(T, D, matrix, PHASE_TAPER) if phase_corr else np.zeros(D.shape[5])
    D = (D * np.exp(-1j * dphi).reshape(1, 1, 1, 1, 1, -1)).astype(np.complex64)
    Dx = bart(1, 'fft -i -u 2', D)                           # readout k -> x (dim 1)
    if Dx is None or np.shape(Dx) != D.shape:
        raise RuntimeError('readout iFFT failed')
    T2 = T[:, :1].copy(); T2[0] = 0                          # (0, k_pe1, k_pe2) per sample
    TI_f = ti_file(scans_plane)
    s_data, s_psf = plane_scaling(Dx, T2, TI_f, matrix, info['noise_sd'], noise_ref_scale)
    cmd = f'moba {opts} --scale_data {s_data:.4f} --scale_psf {s_psf:.6f} -d 4 --img_dims 1:{ny}:{nz}'
    info_slice = dict(matrix=matrix, noise_sd=info['noise_sd'], TI=np.real(TI_f).ravel())
    xs = list(range(nx)) if slices is None else list(slices)
    t0 = time.time()
    with ThreadPoolExecutor(jobs or os.cpu_count()) as ex:
        out = list(ex.map(lambda x: moba_slice(np.ascontiguousarray(Dx[:, x:x + 1]), T2, TI_f, ny, nz, cmd,
                                               dict(info_slice, tag=f'{plane} e{e} x={x}'), progress), xs))
    dt = time.time() - t0
    res = {k: np.stack([o[k] for o in out]) for k in ('maps', 'sens', 'scale_data', 'scale_psf', 'steps',
                                                      'final_res', 'noise_floor')}
    if slices is not None:                                   # embed the subset in a full-size volume
        full = {}
        for k, v in res.items():
            z = np.zeros((nx,) + v.shape[1:], v.dtype)
            z[xs] = v
            full[k] = z
        res = full
    res.update(dphi=dphi, runtime_s=dt, cmd=cmd, slices=np.array(xs))
    res['rel_scale'] = np.where(res['scale_data'] > 0, 0.5 * res['scale_psf'] / np.maximum(res['scale_data'], 1e-12), 0)
    np.savez(f, **res)
    return res, 'reconstructed'


def moba_recon_qc(scans_plane, e, plane, opts, fallback_opts, phantom_dir, proc_dir, flag=1.5, **kw):
    """moba_recon with the residual-QC fallback (tuning stage 1): every readout slice whose raw
    residual exceeds `flag` x its noise floor is reconstructed again with `fallback_opts` (only those
    slices, cached separately) and replaced. Returns (result, state); result['fallback_slices'] lists
    the replaced slices."""
    r, state = moba_recon(scans_plane, e, plane, opts, phantom_dir, proc_dir, **kw)
    r = dict(r)
    bad = np.flatnonzero(r['final_res'] > flag * r['noise_floor'])
    if len(bad):
        rf, _ = moba_recon(scans_plane, e, plane, fallback_opts, phantom_dir, proc_dir, slices=bad, **kw)
        for k in ('maps', 'sens', 'scale_data', 'scale_psf', 'steps', 'final_res', 'noise_floor', 'rel_scale'):
            r[k] = np.array(r[k]); r[k][bad] = rf[k][bad]
    r['fallback_slices'] = bad
    return r, state


def t1_from(maps_list, mask):
    """T1 (s) = 1 / mean over echo groups of R1*; NaN outside `mask` or where R1* <= 0."""
    R1 = np.mean([np.real(m[..., 2]) for m in maps_list], axis=0)
    return np.where(mask & (R1 > 0), 1.0 / np.maximum(R1, 1e-6), np.nan).astype(np.float32)
