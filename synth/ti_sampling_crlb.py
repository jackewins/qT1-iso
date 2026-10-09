"""Which TI carries the most T1 information per unit scan time? Cramér-Rao bound analysis.

    python ti_sampling_crlb.py            # protocol TIs/TRs (TR = TI + 1.2 s, TI31: 1.225 s)

Signed IR signal S = M0 (1 - (1+a) e^{-TI/T1} + a e^{-TR/T1}), white Gaussian noise of equal
variance per k-space sample. A scan with n samples at a given TI contributes n * J^T J to the
Fisher information (J = dS/d(M0, T1[, a])). Scan time is proportional to n * TR, so the TI
allocation is optimised for a fixed total time. Reported: relative CRLB of T1 averaged over brain
tissues (WM 275, GM 340, lesions 192.5 / 357.5 ms), for the current equal-samples protocol and the
optimum with a minimum time share per TI.

Results on 2026-10-02 (see CALIPR_HANDOFF.md): with perfect inversion assumed (the current
2-parameter fit), TI150 carries most information, then TI400; with >= 10-15 % time per TI the optimum
sample shares are ~ TI31 12-18 %, TI150 53-60 %, TI400 18-22 %, TI800 7-11 % (16-20 % lower T1 SD in
the same time). If inversion efficiency must be fitted, the equal-sample protocol is already within
~2 % of optimal and T1 precision is ~2.4x worse - calibrate the inversion efficiency once instead.
"""
import numpy as np
from scipy.optimize import minimize

TI = np.array([30.75, 150.0, 400.0, 800.0])
TR = np.array([1225.0, 1350.0, 1600.0, 2000.0])
TISSUES = {'lesion short': 192.5, 'WM': 275.0, 'GM': 340.0, 'lesion long': 357.5}


def jacobian(T1, a=1.0):
    E, Er = np.exp(-TI / T1), np.exp(-TR / T1)
    S = 1 - (1 + a) * E + a * Er
    dT1 = -(1 + a) * TI / T1 ** 2 * E + a * TR / T1 ** 2 * Er
    da = -E + Er
    return np.stack([S, dT1, da], 1)


def rel_crlb(T1, n, n_par):
    J = jacobian(T1)[:, :n_par]
    F = (J * n[:, None]).T @ J
    return np.sqrt(np.linalg.inv(F)[1, 1]) / T1


def cost(time_share, n_par):
    return np.mean([rel_crlb(t, time_share / TR, n_par) for t in TISSUES.values()])


def optimise(n_par, floor, seeds=20):
    def f(x):
        return cost(floor + (1 - len(TI) * floor) * np.abs(x) / np.abs(x).sum(), n_par)
    best = min((minimize(f, np.random.default_rng(s).random(len(TI)), method='Nelder-Mead',
                         options=dict(xatol=1e-6, fatol=1e-10, maxiter=6000)) for s in range(seeds)),
               key=lambda r: r.fun)
    return floor + (1 - len(TI) * floor) * np.abs(best.x) / np.abs(best.x).sum(), best.fun


if __name__ == '__main__':
    current = TR / TR.sum()                       # equal samples per TI -> time share proportional to TR
    print('T1 sensitivity T1*dS/dT1 per TI (M0 = 1):')
    for k, t in TISSUES.items():
        print(f'  {k:13s}' + ''.join(f'  TI{int(ti):>4d}: {t * d:+.3f}' for ti, d in zip(TI, jacobian(t)[:, 1])))
    for n_par, name in ((2, 'M0, T1 (perfect inversion)'), (3, 'M0, T1, inversion efficiency')):
        c0 = cost(current, n_par)
        print(f'\nmodel {name}: current protocol mean relative CRLB {c0:.2f}')
        for floor in (0.0, 0.10, 0.15):
            w, c = optimise(n_par, floor)
            n = (w / TR) / (w / TR).sum()
            print(f'  min time share {floor:.2f}: sample share TI{[int(t) for t in TI]} = {np.round(n, 3)}  '
                  f'-> {100 * (1 - c / c0):.0f} % lower SD (same precision in {100 * (c / c0) ** 2:.0f} % of the time)')
