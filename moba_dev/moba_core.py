# Prototype of the notebook's reconstruction cells (identical code is pasted into
# Recon_MOBA_V1.ipynb). Run: python moba_core.py <PLANE> [echo ...]
import os, sys, time, json
from pathlib import Path
import numpy as np

REPO = Path('/Users/jackewins/Documents/MRI_PhD/Projects/Blood-and-bleeds/AGLOW_scans/qT1-iso/.claude/worktrees/agent-a675baaf449620218')
sys.path.insert(0, str(REPO / 'synth'))
from phantom import load_scans
import pipeline_utils as pu
from pipeline_utils import bart

PHANTOM_DIR = Path(os.environ.get('QT1_PHANTOM_DIR', '~/work/phantom')).expanduser()
PROC_DIR = PHANTOM_DIR / 'moba'
PROC_DIR.mkdir(exist_ok=True)

MOBA_NEWTON = 8
MOBA_INNER = 100
MOBA_REG = '-l1'
MOBA_ALPHA_MIN = 0.01
MOBA_SCALING = '--normalize_scaling --scale_data 5000 --scale_psf 1000'
PHASE_CORR = True
PHASE_TAPER = 0.15

MOBA_OPTS = f'-L {MOBA_REG} -i {MOBA_NEWTON} -C {MOBA_INNER} -j {MOBA_ALPHA_MIN} {MOBA_SCALING}'
MOBA_TAG = f'L{MOBA_REG}_i{MOBA_NEWTON}_C{MOBA_INNER}_j{MOBA_ALPHA_MIN}_ph{int(PHASE_CORR)}'


def ti_phase_offsets(T, D, matrix, width=PHASE_TAPER, ref=pu.REF_TI_IDX):
    tr = np.real(T)
    w = np.exp(-0.5 * sum((tr[a] / (width * matrix[a])) ** 2 for a in range(3)))[None]
    g = pu.checked(bart(1, 'nufft -a -d {}:{}:{}'.format(*matrix), T, (D * w).astype(np.complex64)))
    prod = np.sum(g * np.conj(g[..., ref])[..., None], axis=3)
    return 0.5 * np.angle(np.sum(prod.astype(np.complex128) ** 2, axis=(0, 1, 2)))


def crop_center(x, shape):
    sl = tuple(slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n, m in zip(x.shape[:3], shape))
    return x[sl]


def moba_recon(scans_plane, e, plane):
    f = PROC_DIR / f'moba_{plane}_e{e}_{MOBA_TAG}.npz'
    if f.exists():
        return dict(np.load(f)), 'cached'
    matrix = scans_plane[pu.REF_TI_IDX]['info']['matrix']
    T, D, P = pu.stack_tis(scans_plane, e)
    dphi = ti_phase_offsets(T, D, matrix) if PHASE_CORR else np.zeros(D.shape[5])
    D = (D * np.exp(-1j * dphi).reshape(1, 1, 1, 1, 1, -1)).astype(np.complex64)
    TI = np.array([s['info']['TI_s'] for s in scans_plane], np.complex64).reshape(1, 1, 1, 1, 1, -1)
    cmd = f'moba {MOBA_OPTS} -d 4 --img_dims {matrix[0]}:{matrix[1]}:{matrix[2]}'
    t0 = time.time()
    x, s = bart(2, cmd, D, TI, t=T)
    dt = time.time() - t0
    if x is None or np.ndim(x) < 3:
        raise RuntimeError('moba failed:\n' + str(bart.stderr)[-2000:])
    x = crop_center(np.squeeze(x), matrix).astype(np.complex64)          # (X, Y, Z, 3)
    s = crop_center(np.squeeze(s), matrix).astype(np.complex64)          # (X, Y, Z, coil)
    log = '\n'.join(l for l in str(bart.stdout).splitlines() if 'Step' in l or 'FISTA' in l or 'Total' in l)
    out = dict(maps=x, sens=s, dphi=dphi, runtime_s=dt, cmd=cmd, log=log)
    np.savez(f, **out)
    return out, 'reconstructed'


if __name__ == '__main__':
    scans = load_scans(PHANTOM_DIR)
    p = sys.argv[1]
    for e in [int(a) for a in sys.argv[2:]] or [0, 1]:
        out, st = moba_recon(scans[p], e, p)
        print(p, e, st, out['runtime_s'], np.degrees(out['dphi']))
        print(out['log'])
