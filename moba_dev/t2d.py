"""2D test bed: one readout position (readout iFFT) of a plane -> moba -> T1 vs truth on that slice."""
import os, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]          # repo root (this file is in moba_dev/)
sys.path.insert(0, str(REPO / 'synth'))
import pipeline_utils as pu
from pipeline_utils import bart
from phantom import load_scans

P = Path(os.environ.get('QT1_PHANTOM_DIR', '~/work/phantom')).expanduser()
plane, x, opts = sys.argv[1], int(sys.argv[2]), sys.argv[3]
e = int(sys.argv[4]) if len(sys.argv) > 4 else 0
sc = load_scans(P)[plane]
truth = dict(np.load(P / f'truth_{plane}.npz'))
mat = sc[-1]['info']['matrix']
T, D, _ = pu.stack_tis(sc, e)
dphi = np.load(P / f'moba/dphi_{plane}_e{e}_w0.15.npy')
D = D * np.exp(-1j * dphi).reshape(1, 1, 1, 1, 1, -1)
Dx = bart(1, 'fft -i -u 2', D.astype(np.complex64))
T2 = T[:, :1].copy(); T2[0] = 0
TI = np.array([s['info']['TI_s'] for s in sc], np.complex64).reshape(1, 1, 1, 1, 1, -1)
t0 = time.time()
xm, s = bart(2, f'moba {opts} --img_dims 1:{mat[1]}:{mat[2]}', np.ascontiguousarray(Dx[:, x:x + 1]), TI, t=T2)
print('time', time.time() - t0)
print('\n'.join(l for l in str(bart.stdout).splitlines() if 'Step' in l or 'FISTA' in l or 'Scaling' in l))
xm = np.squeeze(xm)
sl = tuple(slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n, m in zip(xm.shape[:2], mat[1:]))
xm = xm[sl]
R1 = np.real(xm[..., 2])
lab = truth['label'][x]
for l, n in ((5, 'WM'), (4, 'GM'), (3, 'CSF'), (6, 'les long'), (7, 'les short')):
    m = lab == l
    if m.any():
        print(f'{n:9s} n={m.sum():5d} |Mss| {np.median(np.abs(xm[..., 0])[m]):8.3f} R1 {np.median(R1[m]):7.3f}  T1 {1000/np.median(R1[m]):8.1f} '
              f'ideal {np.median(truth["T1_ideal_ms"][x][m]):7.1f}')
np.save(P / 'moba' / 't2d_last.npy', xm)
