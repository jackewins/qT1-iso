"""Tuning stage 2a (wavelet strength, --l1val) at the stage-1 choice alpha_min = 0.2 with the
residual-QC fallback (flagged slices re-done at alpha_min 0.3). Computes and caches everything that
Recon_MOBA_Tuning_S2.ipynb shows. Run from the repo root:

    python moba_dev/run_stage2.py [--no-planes]

1. screen: 13 readout slices x 2 echo groups for each --l1val in L1_SCREEN;
2. full coronal planes (with the fallback) for L1_PLANES, in that order. l1val = 1 is the stage-1
   alpha_min 0.2 plane (same options; reused from the cache).
--l1val scales only moba's wavelet threshold (moba_conf.l1val -> the wavelet prox); alpha_min, which
also weights the coil penalty and sets stability, stays 0.2.
Progress: <phantom>/moba/stage2.log and moba_progress.log.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'synth'))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import diag_slice as ds  # noqa: E402
import moba_plane as mp  # noqa: E402
import run_stage1 as S1  # noqa: E402
from phantom import load_scans  # noqa: E402

PLANE = S1.PLANE
J, J_FALLBACK = 0.2, 0.3
L1_SCREEN = (0.5, 2, 4, 8)
L1_PLANES = (4, 2, 0.5)          # approved grid (plus 1 = the stage-1 plane), most useful first
FLAG = S1.FLAG


def opts_l1(v, j=J):
    """moba options for wavelet strength v; v = 1 gives exactly the stage-1 option string."""
    return S1.opts_j(j) + ('' if v == 1 else f' --l1val {v}')


def screen(v, scaling):
    o = f'{opts_l1(v)} {scaling}'
    return ds.run_many([dict(plane=PLANE, x=x, opts=o, e=e, tag=f'l1val {v} e{e} x={x}')
                        for e in (0, 1) for x in S1.XS], verbose=False)


def plane(scans, v, proc, e):
    return mp.moba_recon_qc(scans, e, PLANE, opts_l1(v), opts_l1(v, J_FALLBACK), ds.PHANTOM_DIR, proc,
                            flag=FLAG, progress=proc / 'moba_progress.log')


if __name__ == '__main__':
    proc = ds.PHANTOM_DIR / 'moba'
    L = proc / 'stage2.log'
    S1.log(f'stage 2a start, phantom {ds.PHANTOM_DIR} (fingerprint {ds.phantom_fingerprint()})', L)
    scaling = S1.screen_scaling()
    for v in L1_SCREEN:
        t0 = time.time()
        rr = screen(v, scaling)
        q = np.array([r['final_res'] / r['noise_floor'] for r in rr])
        bad = [f"x={r['x']} e{r['e']}" for r in rr if r['final_res'] > FLAG * r['noise_floor']]
        S1.log(f'screen --l1val {v}: flagged {len(bad)}/{len(rr)} {bad}, res/noise median {np.median(q):.2f} '
               f'max {q.max():.2f} ({(time.time() - t0) / 60:.1f} min)', L)
    if '--no-planes' not in sys.argv:
        scans = load_scans(ds.PHANTOM_DIR)[PLANE]
        for v in (1,) + L1_PLANES:
            for e in (0, 1):
                t0 = time.time()
                r, state = plane(scans, v, proc, e)
                q = r['final_res'] / r['noise_floor']
                S1.log(f'plane --l1val {v} echo {e}: {state}, {(time.time() - t0) / 60:.1f} min, fallback slices '
                       f'{r["fallback_slices"].tolist()}, res/noise after fallback median {np.median(q):.2f} max {q.max():.2f}', L)
            S1.log(f'PLANE_DONE --l1val {v}', L)
    S1.log('STAGE2_DONE', L)
