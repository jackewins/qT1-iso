"""Tuning stage 1 (stability boundary, alpha_min): computes and caches everything that
Recon_MOBA_Tuning_S1.ipynb shows. Run from the repo root:

    python moba_dev/run_stage1.py [--planes-only] [--no-planes]

1. screen: 13 readout slices (x = 8..104) x 2 echo groups for each -j in J_SCREEN (single-slice
   harness, cached in <phantom>/moba/diag/cache_<fingerprint>);
2. full coronal planes (planes_to_run): -j 0.3 (V2 baseline), the lowest -j that passes the screen
   (0 slice runs with raw residual > 1.5 x noise floor) and the next grid value below it (cached in
   <phantom>/moba/);
3. a -R 3 screen at the lowest passing -j.
Progress: <phantom>/moba/stage1.log (milestones) and moba_progress.log (one line per slice).
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'synth'))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import diag_slice as ds  # noqa: E402
import moba_plane as mp  # noqa: E402
from phantom import load_scans  # noqa: E402

PLANE = 'COR'
XS = list(range(8, 112, 8))
J_SCREEN = (0.1, 0.15, 0.2, 0.25, 0.3)
J_PLANES = (0.15, 0.2, 0.25)          # candidate grid for full planes (plus 0.3 always)


def planes_to_run(passing):
    """Full planes: 0.3 (V2 baseline), the lowest passing value, and the next grid value below it
    (fails the screen: the case the adaptive fallback is for). At most 3 planes (approved budget)."""
    out = [0.3]
    if passing:
        jl = min(passing)
        out.append(jl)
        below = [j for j in J_SCREEN if j < jl and j in J_PLANES]
        if below:
            out.append(max(below))
    return list(dict.fromkeys(out))


BASE = '-L -l1 -i 10 -C 100 --sobolev_a 220'
FLAG = 1.5


def opts_j(j, extra=''):
    return f'{BASE} -j {j}{extra}'


def screen_scaling(plane=PLANE):
    """The plane's fixed scaling, formatted exactly as moba_plane.moba_recon passes it."""
    Dx, T2, TI_f, info = ds.prepare(plane, 0)
    s_data, s_psf = mp.plane_scaling(Dx, T2, TI_f, info['matrix'], info['noise_sd'])
    return f'--scale_data {s_data:.4f} --scale_psf {s_psf:.6f}'


def screen(j, scaling, extra='', tag=None):
    o = f'{opts_j(j, extra)} {scaling}'
    jobs = [dict(plane=PLANE, x=x, opts=o, e=e, tag=f'{tag or f"j{j}"} e{e} x={x}') for e in (0, 1) for x in XS]
    return ds.run_many(jobs, verbose=False)


def n_flagged(rr):
    return sum(r['final_res'] > FLAG * r['noise_floor'] for r in rr)


def log(msg, f):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f, 'a') as fh:
        fh.write(line + '\n')


if __name__ == '__main__':
    proc = ds.PHANTOM_DIR / 'moba'
    proc.mkdir(exist_ok=True)
    L = proc / 'stage1.log'
    log(f'stage 1 start, phantom {ds.PHANTOM_DIR} (fingerprint {ds.phantom_fingerprint()})', L)
    scaling = screen_scaling()
    log(f'coronal scaling: {scaling}', L)
    passing = []
    if '--planes-only' not in sys.argv:
        for j in J_SCREEN:
            t0 = time.time()
            rr = screen(j, scaling)
            nf = n_flagged(rr)
            q = np.array([r['final_res'] / r['noise_floor'] for r in rr])
            log(f'screen -j {j}: flagged {nf}/{len(rr)}, res/noise median {np.median(q):.2f} max {q.max():.2f} '
                f'({(time.time() - t0) / 60:.1f} min)', L)
            if nf == 0:
                passing.append(j)
    else:
        passing = [j for j in J_SCREEN if n_flagged(screen(j, scaling)) == 0]
    log(f'passing: {passing}', L)
    if '--no-planes' not in sys.argv:
        scans = load_scans(ds.PHANTOM_DIR)[PLANE]
        order = planes_to_run(passing)
        log(f'full planes: {order}', L)
        for j in order:
            for e in (0, 1):
                t0 = time.time()
                r, state = mp.moba_recon(scans, e, PLANE, opts_j(j), ds.PHANTOM_DIR, proc,
                                         progress=proc / 'moba_progress.log')
                q = r['final_res'] / r['noise_floor']
                log(f'plane -j {j} echo {e}: {state}, {(time.time() - t0) / 60:.1f} min, res/noise median '
                    f'{np.median(q):.2f} max {q.max():.2f}, flagged {int((q > FLAG).sum())} {np.flatnonzero(q > FLAG).tolist()}', L)
            log(f'PLANE_DONE -j {j}', L)
    if passing:
        jl = min(passing)
        rr = screen(jl, scaling, extra=' -R 3', tag=f'j{jl} R3')
        log(f'screen -j {jl} -R 3: flagged {n_flagged(rr)}/{len(rr)}', L)
    log('STAGE1_DONE', L)
