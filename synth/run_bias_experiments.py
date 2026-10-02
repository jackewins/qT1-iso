"""Controlled experiments on the lesion-T1 bias (task 1, CALIPR_LesionBias_V1.ipynb).

    python run_bias_experiments.py <cache_dir> COND [COND ...]

COND = <data>:<maps>:<method>:<reg>, e.g. nf:true:CAL:zero
  data    noisy  -> $QT1_PHANTOM_DIR     (default ~/work/phantom, NOISE_RATIO 0.18)
          nf     -> $QT1_PHANTOM_NF_DIR  (default ~/work/phantom_nf, --noise 0; same seed,
                    so the per-TI phases and everything else are identical)
  maps    E2 | E1  ENLIVE shared across TI (pu.shared_selfcal), 2 or 1 map sets, from that data
          true     the phantom's coil maps (truth_<PLANE>.npz['sens'], one set, unit RSS as ENLIVE)
  method  LLR  pu.LLR_CMD (reference, untuned)       CAL  CALIPR V1 (calipr.cal_cmd())
  reg     def  the method's default lambda / iterations
          zero lambda -> 0: CG on the unregularised problem (pics -l2 -r 1e-6), ZERO_ITER
               iterations; for LLR this is per-TI SENSE (no TI coupling left), for CAL the
               subspace-constrained SENSE (the basis still couples the TIs)
          zeroN  as zero with N iterations (convergence check)
          lamX   the method's own regulariser and iterations with lambda = X (dose-response)
          lamXiN the method's own regulariser with lambda = X and N iterations (tuning sweeps)
Each condition is reconstructed for both echo groups, map-set combined, echo-averaged and
fitted exactly as in the V1 notebook / run_llr_reference.py, and cached as
<cache_dir>/<PLANE>_<COND>.npz (img per echo, signed, T1_s, M0, R2, seconds). Existing
results are skipped. Plane: $BIAS_PLANE (default COR).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_utils as pu  # noqa: E402
import calipr as cal  # noqa: E402
from phantom import load_scans  # noqa: E402

PLANE = os.environ.get('BIAS_PLANE', 'COR')
DATA_DIRS = {'noisy': Path(os.environ.get('QT1_PHANTOM_DIR', Path.home() / 'work' / 'phantom')),
             'nf': Path(os.environ.get('QT1_PHANTOM_NF_DIR', Path.home() / 'work' / 'phantom_nf'))}
ZERO_LAMBDA, ZERO_ITER = 1e-6, 100
ECHO_GROUPS = (0, 1)


def parse(cond):
    data, maps, method, reg = cond.split(':')
    assert data in DATA_DIRS and maps in ('E1', 'E2', 'true') and method in ('LLR', 'CAL'), cond
    assert reg == 'def' or reg.startswith('zero') or reg.startswith('lam'), cond
    return data, maps, method, reg


def zero_iters(reg):
    return int(reg[4:]) if len(reg) > 4 else ZERO_ITER


def recon_cmd(method, reg):
    if reg.startswith('zero'):
        return f'pics -S -l2 -r {ZERO_LAMBDA} -i {zero_iters(reg)} -U'
    if reg == 'def':
        return pu.LLR_CMD if method == 'LLR' else cal.cal_cmd()
    lam, _, it = reg[3:].partition('i')
    lam, it = float(lam), int(it) if it else None
    if method == 'LLR':
        cmd = pu.LLR_CMD.replace(f':{pu.LLR_LAMBDA} ', f':{lam} ')
        return cmd if it is None else cmd.replace(f'-i {pu.LLR_ITER} ', f'-i {it} ')
    return cal.cal_cmd(lam=lam) if it is None else cal.cal_cmd(lam=lam, it=it)


def get_sens(cache, data, maps, e, scans_plane):
    if maps == 'true':
        t = np.load(DATA_DIRS[data] / f'truth_{PLANE}.npz')
        return pu.unit_rss(t['sens'])
    f = cache / f'sens_{PLANE}_{data}_{maps}_e{e}.npz'
    if f.exists():
        return np.load(f)['s']
    t0 = time.time()
    s = pu.shared_selfcal(scans_plane, e, nmaps=int(maps[1]))
    np.savez(f, s=s, seconds=time.time() - t0)
    print(f'  maps {data} {maps} e{e}: {time.time() - t0:.0f} s', flush=True)
    return s


def get_dphi(cache, data, maps, e, scans_plane, sens):
    f = cache / f'tiphase_{PLANE}_{data}_{maps}_e{e}.npz'
    if f.exists():
        return np.load(f)['dphi']
    dphi = cal.estimate_ti_phase(scans_plane, sens, e)
    np.savez(f, dphi=dphi)
    return dphi


def run(cache, cond):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    out = cache / f'{PLANE}_{cond.replace(":", "_")}.npz'
    if out.exists():
        print(f'{cond}: cached'); return out
    data, maps, method, reg = parse(cond)
    scans_plane = load_scans(DATA_DIRS[data])[PLANE]
    TI = np.array([s['info']['TI_s'] for s in scans_plane])
    TR = np.array([s['info']['TR_s'] for s in scans_plane])
    truth = np.load(DATA_DIRS[data] / f'truth_{PLANE}.npz')
    t0 = time.time()
    imgs, signed = [], []
    for e in ECHO_GROUPS:
        sens = get_sens(cache, data, maps, e, scans_plane)
        sens = sens if sens.ndim == 5 else sens[..., None]
        cmd = recon_cmd(method, reg)
        if method == 'LLR':
            x = pu.llr_recon(scans_plane, sens, e, cmd=cmd)             # (X,Y,Z,M,TI)
        else:
            dphi = get_dphi(cache, data, maps, e, scans_plane, sens)
            _, x = cal.subspace_recon(scans_plane, sens, e, dphi, cmd=cmd, basis=cal.basis_for(TI, TR))
        x = x if x.ndim == 5 else x[..., None, :]
        imgs.append(x.astype(np.complex64))
        signed.append(np.real(pu.combine_maps(x)))
        print(f'  {cond} e{e}: {time.time() - t0:.0f} s', flush=True)
    S = 0.5 * (signed[0] + signed[1])
    T1, M0, R2 = pu.fit_t1_grid(S, TI, TR, truth['label'] >= 3)
    np.savez(out, img0=imgs[0], img1=imgs[1], signed=S.astype(np.float32), T1_s=T1, M0=M0, R2=R2,
             seconds=time.time() - t0, cond=cond)
    rows = pu.evaluate_t1(T1, truth)
    pu.print_eval(rows, f'\n{PLANE} {cond} ({time.time() - t0:.0f} s)')
    return out


if __name__ == '__main__':
    cache_dir = sys.argv[1]
    for c in sys.argv[2:]:
        run(cache_dir, c)
