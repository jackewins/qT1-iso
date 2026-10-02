"""Figures for comparing T1 reconstructions on one phantom plane (moba, LLR, ...).

All functions take T1 volumes in **ms** on the plane's reconstruction grid (X, Y, Z) and the
plane's truth dict (`truth_<PLANE>.npz`). Colours: the reference truths are neutral (ideal =
grey fill, true = black dashed); compared methods take the categorical slots in fixed order
(validated all-pairs for up to 3 series: blue, orange, aqua). Maps: viridis (T1), RdBu_r
(difference, neutral midpoint).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

SERIES = ('#2a78d6', '#eb6834', '#1baf7a')          # categorical slots 1-3 (fixed order)
IDEAL_FILL, TRUE_INK, GRID = '#bdbdbd', '#0b0b0b', '#e6e6e6'
T1_WIN = (180, 400)                                  # ms; phantom v2: WM 250-300, GM 310-370, lesions 192.5/357.5
DIFF_WIN = 60                                        # ms
TISSUES = ((5, 'WM'), (4, 'GM'), (3, 'CSF'))
LESION_NAMES = ['10 mm long', '6 mm long', '4 mm long', '2 mm long',
                '10 mm short', '6 mm short', '4 mm short', '2 mm short']


def lesion_true_t1(i):
    """True T1 (ms) of lesion i (0-based), from the phantom definition (phantom.TISSUE)."""
    from phantom import LESIONS, TISSUE
    return float(TISSUE[LESIONS[i][0]][1][0])


def _style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def lesion_slices(truth):
    """pe2 index (z) of each lesion's centre and its (x, y) centre."""
    out = []
    for fr in truth['lesion_frac_by_id']:
        z = int(np.argmax(fr.sum(axis=(0, 1))))
        x, y = np.unravel_index(np.argmax(fr[:, :, z]), fr.shape[:2])
        out.append((int(x), int(y), z))
    return out


def t1_gallery(vols, truth, slices, spacing, title='', head=None):
    """Rows = pe2 slices; columns = true, ideal, each method, and each method − ideal."""
    head = truth['label'] >= 3 if head is None else head
    ideal = truth['T1_ideal_ms']
    cols = [('true', truth['T1_ms'], 'map'), ('ideal (resolution-limited)', ideal, 'map')]
    cols += [(k, v, 'map') for k, v in vols.items()]
    cols += [(f'{k} − ideal', v - ideal, 'diff') for k, v in vols.items()]
    asp = spacing[0] / spacing[1]
    fig, axes = plt.subplots(len(slices), len(cols), figsize=(2.6 * len(cols), 2.75 * len(slices) + 1.0),
                             layout='constrained', squeeze=False)
    for r, z in enumerate(slices):
        for c, (name, v, kind) in enumerate(cols):
            ax = axes[r, c]
            kw = dict(cmap='viridis', vmin=T1_WIN[0], vmax=T1_WIN[1]) if kind == 'map' else \
                dict(cmap='RdBu_r', vmin=-DIFF_WIN, vmax=DIFF_WIN)
            ax.imshow(np.where(head[:, :, z], v[:, :, z], np.nan), aspect=asp, interpolation='nearest', **kw)
            ax.contour(truth['lesion_fraction'][:, :, z], [0.25], colors='k', linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(f'pe2 slice z={z}', fontsize=9)
    nm = len(cols) - len(vols)
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(*T1_WIN), 'viridis'), ax=axes[-1, :nm], location='bottom',
                 shrink=0.8, aspect=40, label='T1 (ms)')
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(-DIFF_WIN, DIFF_WIN), 'RdBu_r'), ax=axes[-1, nm:],
                 location='bottom', shrink=0.8, aspect=25, label='difference (ms)')
    fig.suptitle(title + '  (black contours: lesions)', fontsize=11)
    return fig


def orthoviews(vols, truth, spacing, centre=None, title=''):
    """Each volume (rows) in the plane's three orthogonal cuts (columns); true voxel sizes."""
    head = truth['label'] >= 3
    X, Y, Z = truth['label'].shape
    cx, cy, cz = centre or (X // 2 + 8, Y // 2, Z // 2)
    rows = [('ideal', truth['T1_ideal_ms'])] + list(vols.items())
    cuts = [(f'(x, y) at z={cz}: in-plane {spacing[0]:.1f}×{spacing[1]:.1f} mm',
             lambda v: v[:, :, cz], lambda m: m[:, :, cz], spacing[0] / spacing[1]),
            (f'(x, z) at y={cy}: {spacing[0]:.1f}×{spacing[2]:.0f} mm',
             lambda v: v[:, cy, :], lambda m: m[:, cy, :], spacing[0] / spacing[2]),
            (f'(y, z) at x={cx}: {spacing[1]:.1f}×{spacing[2]:.0f} mm',
             lambda v: v[cx, :, :], lambda m: m[cx, :, :], spacing[1] / spacing[2])]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.6 * len(rows)),
                             gridspec_kw=dict(width_ratios=[Y * spacing[1], Z * spacing[2], Z * spacing[2]]))
    for r, (name, v) in enumerate(rows):
        for c, (ctitle, cut, cutm, asp) in enumerate(cuts):
            ax = axes[r, c]
            ax.imshow(np.where(cutm(head), cut(v), np.nan), aspect=asp, cmap='viridis',
                      vmin=T1_WIN[0], vmax=T1_WIN[1], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(ctitle, fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(*T1_WIN), 'viridis'), ax=axes, fraction=0.02, pad=0.01,
                 label='T1 (ms)')
    fig.suptitle(title, fontsize=11)
    return fig


def tissue_masks(truth):
    return {n: binary_erosion(truth['label'] == l) for l, n in TISSUES}


def t1_histograms(vols, truth, title='', ranges=None):
    """Per tissue (eroded masks): histogram of each method's T1 over the ideal (grey fill) and
    the true T1 (black dashed). Medians in the legend."""
    masks = tissue_masks(truth)
    ranges = ranges or {'WM': (150, 420), 'GM': (180, 480), 'CSF': (0, 6000)}
    fig, axes = plt.subplots(1, len(masks), figsize=(5.2 * len(masks), 3.6))
    for ax, (name, m) in zip(axes, masks.items()):
        lo, hi = ranges[name]
        bins = np.linspace(lo, hi, 81)
        ide = truth['T1_ideal_ms'][m]
        ax.hist(np.clip(ide, lo, hi), bins, color=IDEAL_FILL, label=f'ideal (median {np.median(ide):.0f})')
        tr = truth['T1_ms'][m]
        ax.hist(np.clip(tr, lo, hi), bins, histtype='step', color=TRUE_INK, ls='--', lw=1.2,
                label=f'true (median {np.median(tr):.0f})')
        for (k, v), col in zip(vols.items(), SERIES):
            x = v[m]
            x = x[np.isfinite(x)]
            ax.hist(np.clip(x, lo, hi), bins, histtype='step', color=col, lw=2,
                    label=f'{k} (median {np.median(x):.0f}, IQR {np.subtract(*np.percentile(x, [75, 25])):.0f})')
        ax.set_title(f'{name} (eroded mask, n = {m.sum()})', fontsize=10)
        ax.set_xlabel('T1 (ms)' + ('  [clipped to the axis range]' if name == 'CSF' else ''))
        ax.set_ylabel('voxels'); ax.legend(fontsize=7.5, frameon=False); _style(ax)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def lesion_panel(vols, truth, spacing, title='', half=9):
    """Top: each lesion's central slice, zoomed (rows = lesions; columns = true, ideal, methods).
    Bottom: per-voxel T1 inside each lesion mask (>= half its peak partial-volume fraction)."""
    cents = lesion_slices(truth)
    cols = [('true', truth['T1_ms']), ('ideal', truth['T1_ideal_ms'])] + list(vols.items())
    asp = spacing[0] / spacing[1]
    fig = plt.figure(figsize=(2.0 * len(cents) + 1, 2.0 * len(cols) + 4.2))
    gs = fig.add_gridspec(len(cols) + 2, len(cents), height_ratios=[1] * len(cols) + [0.25, 2.2])
    X, Y = truth['label'].shape[:2]
    for c, (x, y, z) in enumerate(cents):
        sx = slice(max(x - half, 0), min(x + half + 1, X)); sy = slice(max(y - half, 0), min(y + half + 1, Y))
        for r, (name, v) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(v[sx, sy, z], cmap='viridis', vmin=T1_WIN[0], vmax=T1_WIN[1], aspect=asp, interpolation='nearest')
            ax.contour(truth['lesion_frac_by_id'][c][sx, sy, z], [0.25], colors='w', linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'{LESION_NAMES[c]}\n(z={z})', fontsize=8.5)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    ax = fig.add_subplot(gs[-1, :])
    w = 0.8 / (len(vols) + 1)
    for i, fr in enumerate(truth['lesion_frac_by_id']):
        m = fr >= 0.5 * fr.max()
        ide = truth['T1_ideal_ms'][m]
        ax.plot([i - 0.42, i + 0.42], [np.mean(ide)] * 2, color=TRUE_INK, lw=1.5, label='ideal (mean)' if i == 0 else None)
        t_true = lesion_true_t1(i)
        ax.plot([i - 0.42, i + 0.42], [t_true] * 2, color=TRUE_INK, lw=1, ls='--', label='true' if i == 0 else None)
        for j, ((k, v), col) in enumerate(zip(vols.items(), SERIES)):
            vals = v[m]
            xpos = i - 0.4 + (j + 1) * w
            ax.scatter(np.full(vals.size, xpos) + np.linspace(-w / 4, w / 4, vals.size), vals, s=10, color=col,
                       alpha=0.6, edgecolors='none', label=k if i == 0 else None)
            ax.plot([xpos - w / 2.5, xpos + w / 2.5], [np.mean(vals)] * 2, color=col, lw=2.5)
    ax.set_xticks(range(len(cents))); ax.set_xticklabels([f'{n}\n(n={int((fr >= 0.5 * fr.max()).sum())})'
                                                         for n, fr in zip(LESION_NAMES, truth['lesion_frac_by_id'])], fontsize=8)
    ax.set_ylabel('T1 (ms)'); ax.set_ylim(100, 450); ax.legend(fontsize=8, frameon=False, ncol=4, loc='upper right')
    ax.set_title('per-voxel T1 in each lesion mask (bars: means)', fontsize=10); _style(ax)
    fig.suptitle(title, fontsize=11)
    return fig


def bias_bars(rows_by_method, title=''):
    """Bias vs ideal (%) per region for each method (rows from pipeline_utils.evaluate_t1)."""
    names = [r['region'] for r in next(iter(rows_by_method.values())) if r['region'] != 'CSF']
    fig, ax = plt.subplots(figsize=(12, 3.8))
    n = len(rows_by_method); w = 0.8 / n
    for j, ((k, rows), col) in enumerate(zip(rows_by_method.items(), SERIES)):
        vals = [r['bias_vs_ideal_pct'] for r in rows if r['region'] != 'CSF']
        xs = np.arange(len(names)) - 0.4 + (j + 0.5) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=k)
        for xx, vv in zip(xs, vals):
            ax.text(xx, vv + (0.6 if vv >= 0 else -0.6), f'{vv:+.1f}', ha='center', va='bottom' if vv >= 0 else 'top',
                    fontsize=6.5, color='#52514e')
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8.5)
    ax.set_ylabel('mean T1 bias vs ideal (%)'); ax.legend(fontsize=8.5, frameon=False); _style(ax)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def sweep_grid(vols_by_setting, truth, slices, spacing, title='', crop=None, diff=False):
    """Rows = settings (e.g. alpha_min), columns = pe2 slices; first row the ideal. Shared window.
    crop: optional (row slice, col slice) zoom; diff=True shows each setting minus the ideal."""
    head = truth['label'] >= 3
    rows = [('ideal', truth['T1_ideal_ms'])] + list(vols_by_setting.items())
    asp = spacing[0] / spacing[1]
    sl = crop or (slice(None), slice(None))
    ph, pw = head[:, :, 0][sl].shape
    hw = (ph * spacing[0]) / (pw * spacing[1])                      # physical height / width of a panel
    pw_in = 2.6 if hw > 0.6 else 3.6
    fig, axes = plt.subplots(len(rows), len(slices), figsize=(pw_in * len(slices), pw_in * hw * len(rows) + 1.0),
                             layout='constrained', squeeze=False)
    for r, (name, v) in enumerate(rows):
        for c, z in enumerate(slices):
            ax = axes[r, c]
            img = np.where(head[:, :, z], v[:, :, z], np.nan)[sl]
            if diff and r > 0:
                img = img - truth['T1_ideal_ms'][:, :, z][sl]
                kw = dict(cmap='RdBu_r', vmin=-DIFF_WIN, vmax=DIFF_WIN)
            else:
                kw = dict(cmap='viridis', vmin=T1_WIN[0], vmax=T1_WIN[1])
            ax.imshow(img, aspect=asp, interpolation='nearest', **kw)
            ax.contour(truth['lesion_fraction'][:, :, z][sl], [0.25], colors='k' if not diff else '#52514e',
                       linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'pe2 slice z={z}', fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(*T1_WIN), 'viridis'), ax=axes[-1, :], location='bottom',
                 shrink=0.6, aspect=40, label='T1 (ms)' if not diff else 'T1 (ms), ideal row')
    if diff:
        fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(-DIFF_WIN, DIFF_WIN), 'RdBu_r'), ax=axes[-1, :],
                     location='bottom', shrink=0.6, aspect=40, label='setting − ideal (ms)')
    fig.suptitle(title, fontsize=11)
    return fig


def slice_qc(results_by_setting, title=''):
    """Raw residual / noise floor per readout slice (both echo groups) for each setting."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for (k, rr), col in zip(results_by_setting.items(), SERIES + ('#e87ba4', '#008300')):
        for e, (r, ls) in enumerate(zip(rr, ('-', ':'))):
            q = r['final_res'] / np.where(r['noise_floor'] > 0, r['noise_floor'], np.nan)
            ax.plot(q, ls, color=col, lw=1.4, label=f'{k}, echo {e}')
    ax.axhline(1, color='#52514e', lw=0.8); ax.axhline(1.5, color='#e34948', lw=0.8, ls='--', label='divergence flag (1.5)')
    ax.set_yscale('log'); ax.set_xlabel('readout slice x'); ax.set_ylabel('raw residual / noise floor')
    ax.legend(fontsize=7, frameon=False, ncol=3); _style(ax); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


SERIES8 = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')  # adjacent-pair order


def hist_by_setting(vols_by_setting, truth, title='', ranges=None):
    """Small multiples: rows = tissues (WM, GM), columns = settings; each panel the setting's T1
    histogram (orange) over the ideal (grey fill) and the true T1 (black dashed)."""
    masks = {k: v for k, v in tissue_masks(truth).items() if k in ('WM', 'GM')}
    ranges = ranges or {'WM': (150, 420), 'GM': (180, 480)}
    n = len(vols_by_setting)
    fig, axes = plt.subplots(len(masks), n, figsize=(3.3 * n, 2.7 * len(masks)), sharey='row', layout='constrained', squeeze=False)
    for r, (tn, m) in enumerate(masks.items()):
        lo, hi = ranges[tn]
        bins = np.linspace(lo, hi, 71)
        ide, tr = truth['T1_ideal_ms'][m], truth['T1_ms'][m]
        for c, (k, v) in enumerate(vols_by_setting.items()):
            ax = axes[r, c]
            x = v[m]; x = x[np.isfinite(x)]
            ax.hist(np.clip(ide, lo, hi), bins, color=IDEAL_FILL, label='ideal')
            ax.hist(np.clip(tr, lo, hi), bins, histtype='step', color=TRUE_INK, ls='--', lw=1, label='true')
            ax.hist(np.clip(x, lo, hi), bins, histtype='step', color=SERIES[1], lw=1.8, label=k)
            ax.set_title(f'{tn}, {k}\nmedian {np.median(x):.0f} (ideal {np.median(ide):.0f}), IQR {np.subtract(*np.percentile(x, [75, 25])):.0f}',
                         fontsize=8.5)
            ax.set_xlabel('T1 (ms)', fontsize=8); _style(ax)
            if c == 0:
                ax.set_ylabel('voxels'); ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    return fig


def truth_histograms(truth, title=''):
    """The phantom's own T1 distributions per tissue (true T1, label masks, not eroded)."""
    from phantom import TISSUE
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout='constrained')
    for ax, (l, name) in zip(axes, ((5, 'WM'), (4, 'GM'))):
        v = truth['T1_ms'][truth['label'] == l]
        lo, hi = TISSUE[l][1]
        ax.hist(v, np.linspace(lo - 20, hi + 20, 61), color=SERIES[0])
        for b in (lo, hi):
            ax.axvline(b, color='#52514e', lw=0.8, ls='--')
        ax.set_title(f'{name}: defined range {lo:.0f}-{hi:.0f} ms; mean {v.mean():.1f}, SD {v.std():.1f} ms '
                     f'({100 * v.std() / (hi - lo):.0f} % of range)', fontsize=9)
        ax.set_xlabel('true T1 (ms)'); ax.set_ylabel('voxels'); _style(ax)
    fig.suptitle(title, fontsize=11)
    return fig
