"""Builds Recon_MOBA_Tuning_S2.ipynb (tuning stage 2a: wavelet strength --l1val at alpha_min 0.2).
Run from the repo root; the heavy runs are done (and cached) by moba_dev/run_stage2.py. The
decision cell (last) is read from report_s2.md."""
import sys
from pathlib import Path

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip('\n')))


md(r"""
# moba `-L` tuning, stage 2a: wavelet strength (`--l1val`) — phantom v2, coronal plane

Stage 2a of the tuning plan, as approved by the researcher after stage 1
(`Recon_MOBA_Tuning_S1.ipynb`): **α_min = 0.2** with the per-slice residual-QC fallback, joint
l1-wavelet regularisation (no TV), and a sweep of the wavelet strength `--l1val`. The stage-1 maps
were judged *very noisy, with some artefacting*; this stage asks how much of the noise a stronger
wavelet removes and at what cost in bias, and characterises the artefacts.

**What `--l1val` does.** In each Newton step FISTA solves
½‖J·dx − r‖² + α_n·(l1val·‖W x_maps‖₁ + ‖ĉ‖²_Sobolev); `--l1val` scales only the joint wavelet
term on the three parameter maps (`moba_conf.l1val` → the wavelet prox). α_n (→ α_min) also weights
the coil penalty and decides stability (stage 1), so the two are decoupled: α_min stays 0.2.
Grid: l1val ∈ {0.5, 1, 2, 4} as full planes (1 = the stage-1 α_min 0.2 result), 8 as a screen.

Everything else as V2/stage 1: `moba -L -l1 -i 10 -C 100 -j 0.2 --sobolev_a 220`, fixed
noise-referenced scaling, slice-wise along the readout, T1 = 1/mean(R1*) over the two echo groups.
Data: **synthetic phantom v2 only**.

| step | what | section |
|---|---|---|
| 1 | screen: 13 readout slices × 2 echo groups per l1val; two failure modes | 3 |
| 2 | full planes: per-slice QC and the fallback | 4 |
| 3 | phantom numbers per l1val (bias vs ideal, noise), histograms | 5 |
| 4 | images for visual assessment | 6 |
| 5 | artefact analysis (noise vs systematic, the centre-slice dip) | 7 |
| 6 | decision | 8 |
""")

md("## 1. Environment")
code(r"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, uniform_filter

os.environ.setdefault('BART_TOOLBOX_PATH', str(Path('~/bart').expanduser()))
REPO = Path.cwd()
sys.path.insert(0, str(REPO / 'synth'))
sys.path.insert(0, str(REPO / 'moba_dev'))
import pipeline_utils as pu
from phantom import load_scans
import diag_slice as ds
import moba_plane as mp
import figures as F
import run_stage1 as S1
import run_stage2 as S2

%matplotlib inline
plt.rcParams['figure.facecolor'] = 'white'
warnings.filterwarnings('ignore', 'All-NaN slice')        # nanmedian over slices without WM
""")

md("## 2. Configuration")
code(r"""
PHANTOM_DIR = ds.PHANTOM_DIR
PROC_DIR = PHANTOM_DIR / 'moba'
PLANE = S2.PLANE
scans = load_scans(PHANTOM_DIR)[PLANE]
SP = pu.spacing_of(scans[pu.REF_TI_IDX]['info'])
truth = dict(np.load(PHANTOM_DIR / f'truth_{PLANE}.npz'))
head = truth['label'] >= 3
SCALING = S1.screen_scaling()
L1_ALL = sorted({1, *S2.L1_SCREEN})
L1_PLANES = sorted({1, *S2.L1_PLANES})
print('phantom', PHANTOM_DIR, '| fingerprint', ds.phantom_fingerprint(), '| alpha_min', S2.J, '| scaling', SCALING)
print('screen l1val', L1_ALL, '| full planes', L1_PLANES)
for v in L1_ALL:
    print(f'  l1val {v}: moba {S2.opts_l1(v)}')
""")

md(r"""
## 3. Screen: two different failure modes

13 readout slices × 2 echo groups per l1val (l1val 1 = the stage-1 screen at α_min 0.2). A slice
run is flagged when its raw residual exceeds 1.5 × its noise floor. The table classifies each
flagged run by two truth-free markers: the largest |M_ss| relative to the plane's typical value
(divergence inflates M_ss, see V2 section 7b) and the median residual of the whole screen.
""")
code(r"""
scr = {v: (S1.screen(S2.J, SCALING) if v == 1 else S2.screen(v, SCALING)) for v in L1_ALL}
fig, ax = plt.subplots(1, 3, figsize=(17, 3.8), layout='constrained')
for (v, rr), col in zip(scr.items(), F.SERIES8):
    for e, ls in ((0, '-'), (1, ':')):
        r_e = [r for r in rr if r['e'] == e]
        x = [r['x'] for r in r_e]
        ax[0].semilogy(x, [r['final_res'] / r['noise_floor'] for r in r_e], 'o' + ls, ms=3, color=col,
                       label=f'l1val {v}' if e == 0 else None)
        ax[1].plot(x, [np.abs(r['maps'][..., 0]).max() for r in r_e], 'o' + ls, ms=3, color=col)
        ax[2].plot(x, [100 * (r['rows']['WM']['med'] / r['rows']['WM']['ideal'] - 1) if 'WM' in r['rows'] else np.nan
                       for r in r_e], 'o' + ls, ms=3, color=col)
ax[0].axhline(1, color='#52514e', lw=0.8); ax[0].axhline(S2.FLAG, color='#e34948', lw=0.8, ls='--')
ax[0].set_ylabel('raw residual / noise floor'); ax[1].set_ylabel('max |M_ss| in the slice'); ax[1].set_yscale('log')
ax[2].set_ylabel('WM median T1 vs ideal (%)'); ax[2].set_ylim(-30, 30); ax[2].axhline(0, color='#52514e', lw=0.8)
for a in ax:
    a.set_xlabel('readout slice x  (solid: echo 0, dotted: echo 1)'); F._style(a)
ax[0].legend(fontsize=8, frameon=False)
fig.suptitle(f'{PLANE}: stage-2a screen (α_min {S2.J}, phantom v2)'); plt.show()
print(f"{'l1val':>6s} {'flagged':>8s} {'res/noise median':>17s} {'max':>8s}  flagged runs (res/noise, max|M_ss| / median of the screen)")
for v, rr in scr.items():
    q = np.array([r['final_res'] / r['noise_floor'] for r in rr])
    mx = np.array([np.abs(r['maps'][..., 0]).max() for r in rr]); med = np.median(mx)
    bad = [f"x={r['x']} e{r['e']} ({qq:.1f}, {m / med:.1f})" for r, qq, m in zip(rr, q, mx) if qq > S2.FLAG]
    print(f'{v:6g} {len(bad):5d}/{len(rr)} {np.median(q):17.2f} {q.max():8.2f}  {bad}')
""")

md(r"""
**Two failure modes.** *Too little regularisation* (l1val 0.5) gives the stage-1 **divergence**:
residual tens to hundreds × the noise floor, M_ss blown up, 20-30 % of brain voxels collapsed, at the
same slices as α_min 0.1 (x = 48-64, 96). *Too much regularisation* (l1val ≥ 2) gives
**over-regularisation** in the low-signal slices at the bottom of the head (x = 16, 24): residual
only 1.5-2 × the noise floor, no collapse, but the wavelet threshold removes the weak signal (M_ss
shrinks to 1-12) and T1 there is far off. At l1val 8 nothing is flagged but the *median* residual
rises to ~1.3 × the noise floor: every slice under-fits, i.e. the regulariser removes signal, not
only noise. The median residual is a usable, truth-free ceiling for the regularisation strength
(discrepancy principle): a well-regularised fit sits at ~0.85-0.95 × the noise floor here.

**Fallback policy (changed from stage 1).** Re-doing a flagged slice at α_min 0.3 is right for
divergence but makes an over-regularised slice worse. Stage 2 therefore replaces every flagged slice
by the **nearest setting towards the validated baseline that passes the QC in that slice**
(l1val 4 → 2 → 1, 2 → 1, 0.5 → 1; the baseline α_min 0.2, l1val 1 converged in all 224 slice runs
in stage 1). This is the discrepancy principle applied per slice on the computed grid: each slice
gets the strongest wavelet in the grid whose fit still explains the data to the noise level. (The
runner also computed the α_min 0.3 re-runs; they are not used.)
""")

md(r"""
## 4. Full planes: per-slice convergence and the fallback

All 112 readout slices × 2 echo groups for each l1val; flagged slices (raw residual > 1.5 × noise
floor, per echo group) are replaced as described above (section 5 also lists the numbers without the
replacement).
""")
code(r"""
raw = {v: [mp.moba_recon(scans, e, PLANE, S2.opts_l1(v), PHANTOM_DIR, PROC_DIR)[0] for e in (0, 1)] for v in L1_PLANES}


def flagged(r):
    return r['final_res'] > S2.FLAG * r['noise_floor']


def with_fallback(v):
    '''Maps of setting v where each flagged slice is taken from the nearest grid setting towards
    l1val 1 that passes the QC there. Returns (maps per echo group, source l1val per slice and echo).'''
    steps = sorted((u for u in L1_PLANES if u != v and min(v, 1) <= u <= max(v, 1)), key=lambda u: abs(np.log(u / v)))
    out, src = [], []
    for e in (0, 1):
        m = raw[v][e]['maps'].copy(); s = np.full(m.shape[0], float(v)); bad = flagged(raw[v][e])
        for u in steps:
            ok = bad & ~flagged(raw[u][e])
            m[ok] = raw[u][e]['maps'][ok]; s[ok] = u; bad &= ~ok
        out.append(m); src.append(s)
    return out, src


def runs(xs):
    '''[13, 15, 16, 17] -> "13, 15-17"'''
    xs, out = list(xs), []
    for x in xs:
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return ', '.join(f'{a}' if a == b else f'{a}-{b}' for a, b in out) or '-'


FB = {v: with_fallback(v) for v in L1_PLANES}
for v in L1_PLANES:
    for e in (0, 1):
        q = raw[v][e]['final_res'] / raw[v][e]['noise_floor']; src = FB[v][1][e]
        repl = ', '.join(f'x = {runs(np.flatnonzero(src == u))} -> l1val {u:g}' for u in sorted(set(src) - {v}))
        print(f"l1val {v:>3g} echo {e}: {float(raw[v][e]['runtime_s']) / 60:5.1f} min, res/noise median {np.median(q):.2f}, "
              f"max {q.max():.2f}; flagged {int(flagged(raw[v][e]).sum()):3d}: {repl or 'none'}")
F.slice_qc({f'l1val {v:g}': raw[v] for v in L1_PLANES}, f'{PLANE}: per-slice convergence, full planes (before fallback)'); plt.show()
""")

md(r"""
## 5. Phantom numbers per l1val

T1 = 1/mean(R1*) of the two echo groups (after the fallback), `pu.evaluate_t1` (eroded tissue masks,
lesion masks ≥ half the peak partial-volume fraction). **Bias vs ideal** is the reconstruction's own
error; **robust SD** (1.4826 × MAD over the eroded mask, ms) measures the noise. The fraction of
the voxel error that is noise is estimated from the two echo groups (independent noise, shared
systematic error). For the 10 and 6 mm lesions (the 4 and 2 mm lesions hold 2 voxels each) the
lesion-WM **contrast** kept by the reconstruction and the voxel **CNR** (contrast / WM robust SD)
show the trade-off in one number each.
""")
code(r"""
T1 = {f'l1val {v:g}': 1000 * mp.t1_from(FB[v][0], head) for v in L1_PLANES}
rows = {k: pu.evaluate_t1(t / 1000, truth) for k, t in T1.items()}
short = lambda k: k.replace('l1val ', 'l1 ')
cols = list(rows)
print(f"{'region':18s} {'n':>5s} {'ideal':>6s}  " + ''.join(f"{short(c):>17s}" for c in cols))
for i, r in enumerate(rows[cols[0]]):
    if r['region'] == 'CSF':
        continue
    print(f"{r['region']:18s} {r['n']:5d} {r['ideal_ms']:6.1f}  "
          + ''.join(f"{rows[c][i]['est_ms']:7.1f} ({rows[c][i]['bias_vs_ideal_pct']:+5.1f}%)" for c in cols))
print('\nwithout the fallback (flagged slices as reconstructed), bias vs ideal:')
for v in L1_PLANES:
    if all((FB[v][1][e] == v).all() for e in (0, 1)):
        continue
    rr = pu.evaluate_t1(mp.t1_from([r['maps'] for r in raw[v]], head), truth)
    print(f'   l1val {v:>3g}: ' + ', '.join(f"{r['region']} {r['bias_vs_ideal_pct']:+.1f}%" for r in rr
                                          if r['region'] in ('WM', 'GM', 'lesion 1 (10 mm)', 'lesion 5 (10 mm)')))
masks = F.tissue_masks(truth)
print()
rsd = {}
for name in ('WM', 'GM'):
    m = masks[name]
    print(f'{name}: ideal median {np.median(truth["T1_ideal_ms"][m]):.1f}')
    for v in L1_PLANES:
        t = T1[f'l1val {v:g}'][m]
        te = [1000 / np.maximum(np.real(FB[v][0][e][..., 2]), 1e-3)[m] for e in (0, 1)]
        noise = 1.4826 * np.median(np.abs(te[0] - te[1])) / 2          # noise of the combined map
        rsd[name, v] = 1.4826 * np.median(np.abs(t - np.median(t)))
        print(f"   l1val {v:>3g}: median {np.median(t):6.1f}, robust SD {rsd[name, v]:5.1f} ms, of which noise ~{noise:5.1f} ms")
print('\nlesion contrast vs WM: (lesion mean − WM mean) as % of the ideal contrast; voxel CNR = |contrast| / WM robust SD')
reg = {r['region']: i for i, r in enumerate(rows[cols[0]])}
LES = [k for k in reg if k.startswith('lesion') and ('10 mm' in k or '6 mm' in k)]
print(f"{'l1val':>6s}  " + ''.join(f'{k:>24s}' for k in LES))
for v in L1_PLANES:
    rr = rows[f'l1val {v:g}']; w = rr[reg['WM']]
    print(f'{v:6g}  ' + ''.join(
        f"{100 * (rr[reg[k]]['est_ms'] - w['est_ms']) / (rr[reg[k]]['ideal_ms'] - w['ideal_ms']):10.0f} %, CNR {abs(rr[reg[k]]['est_ms'] - w['est_ms']) / rsd['WM', v]:4.2f}"
        for k in LES))
F.bias_bars(rows, f'{PLANE}: mean T1 bias vs ideal per l1val (α_min {S2.J}, after fallback)'); plt.show()
F.hist_by_setting(T1, truth, f'{PLANE}: T1 histograms per l1val (eroded masks)'); plt.show()
""")

md(r"""
## 6. Images for visual assessment

Coronal in-plane slices (1.8 × 1.8 mm), rows = l1val (first row the ideal), shared window: the mid
slice and the slices through the lesion centres; the same as differences from the ideal; a zoom on
the lesion band; and the lesion panel.
""")
code(r"""
LZ = sorted(set(z for _, _, z in F.lesion_slices(truth)))
SL = [truth['label'].shape[2] // 2] + LZ
F.sweep_grid(T1, truth, SL, SP, f'{PLANE}: T1 per l1val (α_min {S2.J}, phantom v2)'); plt.show()
F.sweep_grid(T1, truth, SL, SP, f'{PLANE}: T1 − ideal per l1val', diff=True); plt.show()
xs_ = [x for x, _, _ in F.lesion_slices(truth)]
crop = (slice(min(xs_) - 12, max(xs_) + 13), slice(8, truth['label'].shape[1] - 8))
F.sweep_grid(T1, truth, LZ, SP, f'{PLANE}: zoom on the lesion band', crop=crop); plt.show()
sel = {k: T1[k] for k in ('l1val 1', 'l1val 2', 'l1val 4') if k in T1}
F.lesion_panel(sel, truth, SP, f'{PLANE}: lesions per l1val'); plt.show()
""")

md(r"""
## 7. Artefact analysis

The researcher reported noise *and some artefacting*. The two echo groups carry independent noise
but share every systematic error, which separates the two: (a) the error of the combined map,
(b) the noise alone, (echo 0 − echo 1)/2, (c) the systematic part, (a) smoothed, (d) the per-voxel
errors of the two echo groups against each other, (e) the WM error per pe2 slice and (f) per (y, z)
averaged along the readout — anything repeated in every readout slice shows up there as a line.
Shown for l1val 1 (the stage-1 result) and the strongest stable setting.
""")
code(r"""
brain = binary_erosion(truth['label'] >= 4); wm = binary_erosion(truth['label'] == 5); ide = truth['T1_ideal_ms']


def sm(a, k=(5, 5, 1)):
    m = np.isfinite(a); den = uniform_filter(m.astype(float), k)
    return np.where(den > 0.5, uniform_filter(np.where(m, a, 0), k) / np.maximum(den, 1e-9), np.nan)


def per_z(Tmap):
    E = np.where(wm, Tmap - ide, np.nan)
    with np.errstate(all='ignore'):
        return np.nanmedian(E.transpose(2, 0, 1).reshape(E.shape[2], -1), axis=1)


for v in (1, max(L1_PLANES)):
    maps = FB[v][0]
    Te = [1000 / np.maximum(np.real(m[..., 2]), 1e-3) for m in maps]
    Tm = T1[f'l1val {v:g}']
    e0, e1 = (Te[0] - ide)[brain], (Te[1] - ide)[brain]
    ok = (np.abs(e0) < 400) & (np.abs(e1) < 400)
    h0 = np.abs(e0) > 100
    print(f'l1val {v:g}: echo-0/echo-1 voxel error correlation {np.corrcoef(e0[ok], e1[ok])[0, 1]:.2f}; '
          f'voxels with |error| > 100 ms in echo 0: {100 * h0.mean():.1f} %, of which also in echo 1 (same sign): '
          f'{100 * np.mean((np.abs(e1[h0]) > 100) & (np.sign(e1[h0]) == np.sign(e0[h0]))):.0f} % '
          f'(chance {100 * np.mean(np.abs(e1) > 100):.0f} %); too long: {100 * np.mean(e0[h0] > 0):.0f} %')
    z = 22
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), layout='constrained')
    for a, (img, t, lim) in zip(ax[:3], ((Tm - ide, '(a) T1 − ideal', 80), ((Te[0] - Te[1]) / 2, '(b) noise: (echo 0 − echo 1)/2', 80),
                                         (sm(np.where(brain, Tm - ide, np.nan)), '(c) systematic: (a) smoothed 5×5', 30))):
        im = a.imshow(np.where(brain[:, :, z], img[:, :, z], np.nan), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect=SP[0] / SP[1])
        a.set_title(f'l1val {v:g}, slice z={z}: {t}', fontsize=9); a.axis('off'); plt.colorbar(im, ax=a, fraction=0.046, label='ms')
    ax[3].hexbin(e0[ok], e1[ok], gridsize=60, cmap='Greys', bins='log', extent=(-200, 300, -200, 300))
    ax[3].set_xlabel('echo 0: T1 − ideal (ms)'); ax[3].set_ylabel('echo 1: T1 − ideal (ms)'); F._style(ax[3])
    ax[3].set_title(f'(d) per-voxel errors, echo 0 vs echo 1 (corr {np.corrcoef(e0[ok], e1[ok])[0, 1]:.2f})', fontsize=9)
    plt.show()

fig, ax = plt.subplots(1, 2, figsize=(17, 3.8), layout='constrained')
for v, col in zip(L1_PLANES, F.SERIES8):
    ax[0].plot(per_z(T1[f'l1val {v:g}']), 'o-', ms=3, color=col, label=f'l1val {v:g}')
ax[0].axvline(22, color='#52514e', lw=0.8, ls=':'); ax[0].axhline(0, color='#52514e', lw=0.8)
ax[0].set_xlabel('pe2 slice z (centre z = 22)'); ax[0].set_ylabel('median WM T1 − ideal (ms)'); ax[0].set_ylim(-30, 20)
ax[0].legend(fontsize=8, frameon=False); F._style(ax[0]); ax[0].set_title('(e) WM error per pe2 slice', fontsize=9)
E = np.where(wm, T1['l1val 1'] - ide, np.nan)
with np.errstate(all='ignore'):
    Eyz = np.nanmedian(E, axis=0)
im = ax[1].imshow(np.where(np.sum(np.isfinite(E), 0) >= 15, Eyz, np.nan), cmap='RdBu_r', vmin=-30, vmax=30, aspect='auto')
ax[1].set_xlabel('pe2 slice z'); ax[1].set_ylabel('pe1 column y'); plt.colorbar(im, ax=ax[1], fraction=0.03, label='ms')
ax[1].set_title('(f) l1val 1: WM error averaged along the readout, per (y, z)', fontsize=9)
plt.show()
ctr, oth = np.r_[21:24], np.r_[10:19, 26:34]
for v in L1_PLANES:
    pz = per_z(T1[f'l1val {v:g}'])
    Te = [1000 / np.maximum(np.real(m[..., 2]), 1e-3) for m in FB[v][0]]
    nz_ = [1.4826 * np.median(np.abs((Te[0] - Te[1])[:, :, z][wm[:, :, z]])) / 2 if wm[:, :, z].sum() > 200 else np.nan
           for z in range(wm.shape[2])]
    print(f'l1val {v:>3g}: WM error at z=21-23 vs the other slices: {np.nanmean(pz[ctr]) - np.nanmedian(pz[oth]):+5.1f} ms; '
          f'WM noise at z=21-23 {np.nanmean(np.take(nz_, ctr)):.1f} ms vs {np.nanmedian(np.take(nz_, oth)):.1f} ms elsewhere')
""")

md(r"""
**What the artefacts are.**

1. **Bright single voxels (speckle) are noise, not artefacts.** The voxel errors of the two echo
   groups are nearly uncorrelated, and voxels far off in one echo group are far off in the other
   only at chance level. Most of them are too *long*: T1 = 1/R1*, so symmetric noise in R1* becomes
   a long-T1 tail. They shrink with stronger regularisation (sections 5-6) and would average out
   further with more data (e.g. the 3-plane combination).
2. **A dip in the central coronal slices (z = 21-23, WM 7-15 ms low)**, the same in every α_min and
   l1val setting and on both phantom versions; LLR shows a smaller one (−3.6 ms). It has a
   slice-wide part (−7 ms even > 30 mm from the ventricles) and a stronger local part near the
   ventricles / deep GM. z = 22 is the centre of the pe2 axis, where the coarsest level of moba's
   dyadic wavelet splits its 2× oversampled grid, and moba uses the same fixed sequence of random
   wavelet shifts in every readout slice (`wavthresh_rand_state_set(prox, 1)` in
   `src/moba/iter_l1.c`), so a grid-locked error repeats in all 112 slices. **Hypothesis, not yet
   tested:** the proposed test shifts the object by two pe2 slices before reconstruction (a k-space
   phase ramp) and checks whether the dip moves with the grid or with the anatomy.
3. **Lines along the readout near the deep GM / ventricles** (panel f): regularisation smearing
   across those structures, repeated in every readout slice because each readout slice is an
   independent 2D problem with the same geometry.
4. Horizontal streaks outside the brain (scalp level, background): each readout slice is
   reconstructed independently; there is no regularisation along the readout. Not in brain tissue.
""")

REPORT = Path(__file__).with_name('report_s2.md')
md(REPORT.read_text() if REPORT.exists() else '## 8. Decision\n\n*(written after the runs)*')

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}
out = sys.argv[1] if len(sys.argv) > 1 else 'Recon_MOBA_Tuning_S2.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
