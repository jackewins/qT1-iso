"""CALIPR-style subspace reconstruction: the method-specific building blocks of
Recon_CALIPR_V1.ipynb (sections 4, 6 and 7), unchanged, as functions so that the
diagnostic and tuning notebooks call exactly the same code.

Everything shared with the other methods (loading, stacking, coil maps, map-set
combination, T1 fit, evaluation) stays in pipeline_utils.py.

Model: x_TI(r) = sum_k Phi[TI, k] c_k(r), solved with
    pics -B <basis> -R W:7:0:<lambda> ...
Basis layout for pics -B: (1, 1, 1, 1, 1, TI, K) - TI in dim 5 (TE_DIM), K in dim 6 (COEFF_DIM).
"""
import numpy as np

import pipeline_utils as pu
from pipeline_utils import bart

# V1 defaults (Recon_CALIPR_V1.ipynb, untuned)
DICT_T1_S = (0.1, 4.0)
DICT_N = 1000
DICT_UNIT_NORM = True
BASIS_K = 3
PHASE_EST_CMD = 'pics -S -l2 -r 0.01 -i 20 -U'
CAL_LAMBDA, CAL_ITER = 0.005, 80


def cal_cmd(lam=CAL_LAMBDA, it=CAL_ITER, reg='W:7:0', extra=''):
    """pics command for the subspace reconstruction (t=, p=, B= go in as wrapper kwargs)."""
    return f'pics -e -S -R {reg}:{lam} -i {it} -U{extra}'


def ir_dictionary(TI, TR, t1_range_s=DICT_T1_S, n=DICT_N, unit_norm=DICT_UNIT_NORM):
    """Signed IR curves at the protocol's (TI, TR) pairs. Returns T1 grid, raw and fitted dictionary."""
    T1 = np.logspace(*np.log10(t1_range_s), n)
    D = np.stack([pu.ir_signal(t, TI, TR) for t in T1])            # (n, TI)
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True) if unit_norm else D
    return T1, D, Dn


def make_basis(Dn, K):
    """SVD of the dictionary; the first K right singular vectors, real, (TI, K)."""
    _, sv, Vh = np.linalg.svd(Dn, full_matrices=False)
    V = Vh[:K].T
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(K)])   # fix SVD sign (cosmetic)
    return V, sv


def to_bart_basis(V):
    """(TI, K) -> BART basis (1, 1, 1, 1, 1, TI, K)."""
    return V.reshape(1, 1, 1, 1, 1, *V.shape).astype(np.complex64)


def basis_for(TI, TR, K=BASIS_K, **kw):
    _, _, Dn = ir_dictionary(TI, TR, **kw)
    V, _ = make_basis(Dn, K)
    return to_bart_basis(V)


def estimate_ti_phase(scans_plane, sens, e, cmd=PHASE_EST_CMD, ref=pu.REF_TI_IDX):
    """Global phase of each TI relative to the reference TI (rad), from per-TI SENSE images.
    Sign-free (squared), magnitude-weighted correlation with the reference TI (V1 section 6)."""
    xs = []
    for s in scans_plane:
        t, d = pu.load_echo(s['stem'], e)
        x = pu.checked(bart(1, cmd, d, sens, t=t))
        xs.append(x if x.ndim == 4 else x[..., None])                 # (X, Y, Z, map set)
    x = np.stack(xs, axis=-1)                                          # (X, Y, Z, M, TI)
    z = x * np.conj(x[..., ref])[..., None]
    w = np.sum(z ** 2 / np.maximum(np.abs(z), 1e-30), axis=(0, 1, 2, 3))
    return (0.5 * np.angle(w)).astype(np.float64)


def demodulate(D, dphi):
    """Remove a global phase per TI (dim 5) from stacked k-space."""
    return (D * np.exp(-1j * np.asarray(dphi)).reshape((1,) * 5 + (-1,))).astype(np.complex64)


def expand(c, basis):
    """Coefficients (X,Y,Z,1,M,1,K) -> TI images (X,Y,Z,1,M,TI) with fmac (sum over dim 6)."""
    x = bart(1, 'fmac -s 64', c, basis)
    return x.reshape(x.shape + (1,) * (6 - x.ndim))


def subspace_recon(scans_plane, sens, e, dphi, cmd=None, basis=None, stacked=None):
    """Subspace reconstruction of one plane / echo group (V1 section 7).
    sens: (X,Y,Z,coil[,maps]). stacked: optional (T, D, P) from pu.stack_tis (or a replacement).
    Returns coefficients (X,Y,Z,M,K) and TI images (X,Y,Z,M,TI)."""
    cmd = cmd or cal_cmd()
    T, D, P = stacked if stacked is not None else pu.stack_tis(scans_plane, e)
    D = demodulate(D, dphi)
    kw = {'B': basis} if T is None else {'t': T, 'B': basis}
    if P is not None:
        kw['p'] = P
    sens = sens if sens.ndim == 5 else sens[..., None]
    c = bart(1, cmd, D, sens, **kw)
    if c is None or np.ndim(c) < 3:
        raise RuntimeError('pics -B failed - see the error printed above')
    c = c.reshape(c.shape + (1,) * (7 - c.ndim))                       # (X,Y,Z,1,M,1,K)
    assert c.shape[5] == 1 and c.shape[6] == basis.shape[6], c.shape
    x = expand(c, basis)                                               # (X,Y,Z,1,M,TI)
    assert x.shape[5] == basis.shape[5], x.shape
    return c[:, :, :, 0, :, 0, :].astype(np.complex64), x[:, :, :, 0, :, :].astype(np.complex64)
