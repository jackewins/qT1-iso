"""Does `pics -B` honour sensitivity maps that vary along the TI/scan dimension (dim 5)?

    python check_pics_basis_sensdim.py

Toy problem: 3 "scans" whose coil maps differ by a per-scan phase ramp, a K = 2 temporal basis,
fully sampled Cartesian data. Result with BART v1.0.00 (2026-10-08):

    per-scan maps, no basis   -> exact (rel. error ~2e-7)
    per-scan maps, with -B    -> rel. error 0.37, IDENTICAL to using scan 0's maps for all scans

i.e. with a subspace basis pics silently uses one set of maps along dim 5. Consequence for the
multi-plane CALIPR-SR design: maps must be shared by all scans; per-scan phase must be removed
elsewhere (constant phase: demodulate k-space; linear phase: shift the trajectory), or the forward
operator must be written outside pics. Re-run this check after a BART upgrade.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.environ.get('BART_TOOLBOX_PATH', '/usr/local/bart'), 'python'))
from bart import bart  # noqa: E402


def rel(a, b):
    a = a.reshape(b.shape)
    s = np.vdot(a, b) / np.vdot(a, a)
    return float(np.linalg.norm(s * a - b) / np.linalg.norm(b))


rng = np.random.default_rng(0)
N, C, S, K = 24, 4, 3, 2
B = np.linalg.qr(rng.normal(size=(S, K)))[0].astype(np.complex64)
coef = (rng.normal(size=(N, N, 1, 1, 1, 1, K)) + 1j * rng.normal(size=(N, N, 1, 1, 1, 1, K))).astype(np.complex64)
img = np.einsum('xyzabck,sk->xyzabs', coef, B).reshape(N, N, 1, 1, 1, S)
yy, xx = np.mgrid[:N, :N] / N
sens = np.zeros((N, N, 1, C, 1, S), np.complex64)
for c in range(C):
    for s in range(S):
        sens[:, :, 0, c, 0, s] = np.exp(-((xx - c / C) ** 2 + (yy - 0.5) ** 2) * 3) * np.exp(1j * (0.8 * s * xx + c))
sens /= np.sqrt((np.abs(sens) ** 2).sum(3, keepdims=True))
ksp = bart(1, 'fft -u 3', (img * sens).astype(np.complex64))
basis = B.reshape(1, 1, 1, 1, 1, S, K)
cmd = 'pics -S -l2 -r 1e-6 -i 100'
print('per-scan maps, no basis :', rel(bart(1, cmd, ksp, sens), img))
print('scan-0 maps,   no basis :', rel(bart(1, cmd, ksp, sens[..., :1]), img), '(expected to be wrong)')
print('per-scan maps, with -B  :', rel(bart(1, cmd, ksp, sens, B=basis), coef))
print('scan-0 maps,   with -B  :', rel(bart(1, cmd, ksp, sens[..., :1], B=basis), coef))
print('-> if the two -B lines are identical, pics -B ignores per-scan maps')
