"""Reference method on the phantom: python run_llr_reference.py <phantom_dir> [PLANE ...]

Shared-TI self-calibrated coil maps (2 map sets) -> joint LLR of the 4 TIs (LLR_CMD) for
each echo group -> phase-referenced map-set combination -> signed signal averaged over the
two echo groups -> closed-form grid T1 fit -> comparison with the truth. Writes
<phantom_dir>/llr_<PLANE>.npz and llr_<PLANE>.png, and prints the evaluation table.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phantom import load_scans  # noqa: E402
import pipeline_utils as pu  # noqa: E402

out = Path(sys.argv[1])
planes = sys.argv[2:] or ['COR']
scans = load_scans(out)

for p in planes:
    t0 = time.time()
    sp = scans[p]
    signed = []
    for e in range(2):
        sens = pu.shared_selfcal(sp, e, nmaps=2)
        x = pu.llr_recon(sp, sens, e)                       # (X, Y, Z, M, TI)
        signed.append(np.real(pu.combine_maps(x)))
    S = 0.5 * (signed[0] + signed[1])
    TI = np.array([s['info']['TI_s'] for s in sp]); TR = np.array([s['info']['TR_s'] for s in sp])
    truth = np.load(out / f'truth_{p}.npz')
    mask = truth['label'] >= 3                              # CSF, GM, WM, lesions
    T1, M0, R2 = pu.fit_t1_grid(S, TI, TR, mask)
    rows = pu.evaluate_t1(T1, truth)
    pu.print_eval(rows, f'\n{p}: LLR reference ({pu.LLR_CMD}), {time.time() - t0:.0f} s')
    np.savez_compressed(out / f'llr_{p}.npz', signed=S.astype(np.float32), T1_s=T1, M0=M0, R2=R2)

    z = T1.shape[2] // 2
    info = sp[-1]['info']; asp = pu.spacing_of(info)[0] / pu.spacing_of(info)[1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    for a, v, ttl, kw in ((ax[0], 1000 * T1[:, :, z], 'LLR T1 (ms)', dict(vmin=150, vmax=420)),
                          (ax[1], truth['T1_ms'][:, :, z], 'true T1 (ms)', dict(vmin=150, vmax=420)),
                          (ax[2], 1000 * T1[:, :, z] - truth['T1_ms'][:, :, z], 'LLR - true (ms)',
                           dict(vmin=-80, vmax=80, cmap='RdBu_r'))):
        im = a.imshow(v, aspect=asp, **({'cmap': 'viridis'} | kw))
        a.set_title(f'{p} slice {z}: {ttl}'); a.axis('off'); plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout(); plt.savefig(out / f'llr_{p}.png', dpi=70); plt.close()
