"""Sanity checks for a generated phantom: python check_phantom.py <out_dir>

Writes three figures to <out_dir>:
  check_truth.png     ground-truth T1 and labels (1.8 mm iso grid, three orthogonal views)
  check_gridding.png  adjoint-NUFFT (gridding) magnitude of each plane's TI800, coil RSS
  check_overlay.png   the three planes' gridding images resampled to one scanner-frame grid
                      (R = AX, G = COR, B = SAG); they must overlap, since there is no motion
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phantom import AXIS, load_scans  # noqa: E402
from pipeline_utils import bart, load_echo, spacing_of  # noqa: E402

out = Path(sys.argv[1])
scans = load_scans(out)

# ---- truth ---------------------------------------------------------------------------
t = np.load(out / 'truth_iso.npz')
c = t['T1_ms'].shape[0] // 2
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for j, (name, cut) in enumerate([('axial', lambda v: v[c + 20]), ('coronal', lambda v: v[:, :, c]),
                                 ('sagittal', lambda v: v[:, c - 17])]):
    im = ax[0, j].imshow(cut(t['T1_ms']), cmap='viridis', vmin=150, vmax=420)
    ax[0, j].set_title(f'true T1 (ms), {name}'); plt.colorbar(im, ax=ax[0, j], fraction=0.046)
    ax[1, j].imshow(cut(t['label']), cmap='tab10', vmin=0, vmax=9)
    ax[1, j].contour(cut(t['lesion_id']) > 0, [0.5], colors='w', linewidths=0.8)
    ax[1, j].set_title(f'labels (white = lesions), {name}')
for a in ax.ravel():
    a.axis('off')
plt.tight_layout(); plt.savefig(out / 'check_truth.png', dpi=70); plt.close()

# ---- gridding per plane --------------------------------------------------------------
grid = {}
for p, sl in scans.items():
    s = sl[-1]
    tr, d = load_echo(s['stem'], 0)
    im = np.squeeze(bart(1, 'nufft -a -d {}:{}:{}'.format(*s['info']['matrix']), tr, d))
    grid[p] = (np.sqrt((np.abs(im) ** 2).sum(-1)), s['info'])
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
for a, (p, (v, info)) in zip(ax, grid.items()):
    sp = spacing_of(info)
    a.imshow(v[:, :, v.shape[2] // 2], cmap='gray', aspect=sp[0] / sp[1])
    a.set_title(f'{p} TI800 gridding, mid pe2 slice'); a.axis('off')
plt.tight_layout(); plt.savefig(out / 'check_gridding.png', dpi=70); plt.close()


# ---- overlay in the scanner frame ------------------------------------------------------
def to_iso(v, info, n=128, dx=1.8):
    """Sample a plane's volume at the scanner-frame iso grid (linear interpolation)."""
    ax_ = (np.arange(n) - n // 2) * dx
    X, Y, Z = np.meshgrid(ax_, ax_, ax_, indexing='ij')
    xyz = [X, Y, Z]
    sp = spacing_of(info)
    idx = [xyz[AXIS[a]] / sp[j] + info['matrix'][j] // 2 for j, a in enumerate(info['geom'])]
    return map_coordinates(v, idx, order=1, cval=0.0)


iso = {p: to_iso(v / np.percentile(v, 99.5), info) for p, (v, info) in grid.items()}
c = 64
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, (name, cut) in zip(ax, [('axial', lambda v: v[c + 20]), ('coronal', lambda v: v[:, :, c]),
                               ('sagittal', lambda v: v[:, c - 17])]):
    a.imshow(np.clip(np.stack([cut(iso[p]) for p in ('AX', 'COR', 'SAG')], -1), 0, 1))
    a.set_title(f'{name}: R = AX, G = COR, B = SAG'); a.axis('off')
plt.tight_layout(); plt.savefig(out / 'check_overlay.png', dpi=70); plt.close()
print('wrote check_truth.png, check_gridding.png, check_overlay.png to', out)
