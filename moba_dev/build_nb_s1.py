"""Builds Recon_MOBA_Tuning_S1.ipynb (tuning stage 1: the stability boundary in alpha_min).
Run from the repo root; the heavy runs are done (and cached) by moba_dev/run_stage1.py, the
notebook recomputes anything missing. The decision cell (last) is read from report_s1.md."""
import sys
from pathlib import Path

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip('\n')))


md(r"""
# moba `-L` tuning, stage 1: the stability boundary (α_min) — phantom v2, coronal plane

Stage 1 of the tuning plan in `Recon_MOBA_V2.ipynb` (section 18), approved by the researcher:
**how low can the minimum regularisation α_min (`-j`) go before Newton steps diverge?** V2 uses
0.3 conservatively (0.1 diverged in 6 of 26 screened slice runs on phantom v1). Lower α_min
means less smoothing, so less lesion/GM bias and more noise.

Everything else is V2: `moba -L -l1 -i 10 -C 100 --sobolev_a 220`, fixed noise-referenced
scaling (`--scale_data 4.545/σ`, `--scale_psf` = moba's PSF normalisation), slice-wise along the
readout, echo groups combined as T1 = 1/mean(R1*). Data: **synthetic phantom only**.

**Phantom v2** (`recon-comparison` f2840be, merged here): WM 250-300 ms, GM 310-370 ms (bell-
shaped within tissue), lesions 357.5 / 192.5 ms (±30 % of the WM mid-range 275 ms), and centred
partial-volume truth maps (in v1 the lesion evaluation masks were offset by 0.375 voxel through-
plane). All numbers here are on v2 and are not comparable one-to-one with V2's (v1) numbers.

| step | what | section |
|---|---|---|
| 0 | phantom v2: the tissue T1 distributions | 3 |
| 1 | screen: α_min ∈ {0.1, 0.15, 0.2, 0.25, 0.3}, 13 readout slices × 2 echo groups | 4 |
| 2 | full coronal planes for 0.3 and every value that passes, per-slice convergence | 5 |
| 3 | phantom numbers per α_min (bias vs ideal), T1 histograms | 6 |
| 4 | images for visual assessment: T1 maps, differences, lesion zooms | 7 |
| 5 | adaptive fallback; `-R 3` | 8 |
| 6 | what these maps are (geometry) and the decision | 9-10 |

**Decision rule (agreed in the plan):** the smallest α_min with no flagged slice run (raw residual >
1.5 × noise floor) on the full plane, plus one grid step of margin — or the adaptive fallback if
the boundary is slice-specific and a lower α_min clearly reduces lesion bias. The researcher
judges the images.
""")

md("## 1. Environment")
code(r"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

os.environ.setdefault('BART_TOOLBOX_PATH', str(Path('~/bart').expanduser()))
REPO = Path.cwd()
sys.path.insert(0, str(REPO / 'synth'))
sys.path.insert(0, str(REPO / 'moba_dev'))
import pipeline_utils as pu
from phantom import load_scans, TISSUE, LESIONS
import diag_slice as ds              # single-readout-slice harness (cached)
import moba_plane as mp              # full-plane slice-wise moba (cached)
import figures as F                  # comparison figures
import run_stage1 as S1              # the stage-1 settings (same as the runner)

%matplotlib inline
plt.rcParams['figure.facecolor'] = 'white'
""")

md(r"""
## 2. Configuration

The grid and options come from `moba_dev/run_stage1.py`, which ran the heavy parts in the
background; every result is cached (single slices in `<phantom>/moba/diag/cache_<fingerprint>`,
full planes in `<phantom>/moba/`), keyed by the options and by the **phantom fingerprint** (a
hash of `phantom_config.json`), so a different phantom version can never reuse them.
""")
code(r"""
PHANTOM_DIR = ds.PHANTOM_DIR
PROC_DIR = PHANTOM_DIR / 'moba'
PLANE = S1.PLANE
scans = load_scans(PHANTOM_DIR)[PLANE]
info = scans[pu.REF_TI_IDX]['info']
SP = pu.spacing_of(info); ASP = SP[0] / SP[1]
truth = dict(np.load(PHANTOM_DIR / f'truth_{PLANE}.npz'))
head = truth['label'] >= 3
SCALING = S1.screen_scaling()
print('phantom', PHANTOM_DIR, '| fingerprint', ds.phantom_fingerprint(), '| plane', PLANE, info['matrix'], 'voxel (mm)',
      tuple(round(s, 2) for s in SP))
print('tissues (ms):', {TISSUE[l][0]: TISSUE[l][1] for l in (3, 4, 5, 6, 7)})
print('base options:', S1.BASE, '| scaling:', SCALING, '| screen alpha_min:', S1.J_SCREEN)
""")

md(r"""
## 3. Phantom v2: tissue T1 distributions

The researcher's correction: in healthy tissue both WM and GM follow a bell-shaped distribution,
WM within 250-300 ms and GM within 310-370 ms. Phantom v2 keeps the smooth spatial T1 field of
v1 and maps it onto these ranges, which gives a bell with SD ≈ 15 % of the range (bounds at
≈ ±3 SD). **Note:** when asked how the bounds should relate to the bell, the researcher chose
"bounds = ±2 SD" (SD = range/4, WM 12.5 ms, GM 15 ms); v2 as pushed is narrower (below). All
methods must use one phantom, so this stage uses v2 as pushed; a change belongs on
`recon-comparison`.
""")
code(r"""
F.truth_histograms(truth, f'{PLANE} phantom v2: true T1 per tissue (label masks)'); plt.show()
""")

md(r"""
## 4. Screen: 13 readout slices × 2 echo groups per α_min

Single readout slices x = 8, 16, …, 104 (bottom to top of the head), both echo groups, each
α_min. A slice run is **flagged** when its raw-data residual exceeds 1.5 × its noise floor
(V2 section 7: converged fits sit at 0.85-0.9 ×, diverged ones at 3-90 ×).
""")
code(r"""
scr = {j: S1.screen(j, SCALING) for j in S1.J_SCREEN}


def plot_screen(scr, title):
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.8), layout='constrained')
    for (j, rr), col in zip(scr.items(), F.SERIES8):
        for e, ls in ((0, '-'), (1, ':')):
            r_e = [r for r in rr if r['e'] == e]
            x = [r['x'] for r in r_e]
            ax[0].semilogy(x, [r['final_res'] / r['noise_floor'] for r in r_e], 'o' + ls, ms=3, color=col,
                           label=f'α_min {j}' if e == 0 else None)
            ax[1].plot(x, [100 * r['rows']['collapsed_frac'] for r in r_e], 'o' + ls, ms=3, color=col)
            ax[2].plot(x, [100 * (r['rows']['WM']['med'] / r['rows']['WM']['ideal'] - 1) if 'WM' in r['rows'] else np.nan
                           for r in r_e], 'o' + ls, ms=3, color=col)
    ax[0].axhline(1, color='#52514e', lw=0.8); ax[0].axhline(S1.FLAG, color='#e34948', lw=0.8, ls='--')
    ax[0].set_ylabel('raw residual / noise floor'); ax[1].set_ylabel('collapsed brain voxels (%)')
    ax[2].set_ylabel('WM median T1 vs ideal (%)'); ax[2].set_ylim(-30, 30); ax[2].axhline(0, color='#52514e', lw=0.8)
    for a in ax:
        a.set_xlabel('readout slice x  (solid: echo 0, dotted: echo 1)'); F._style(a)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title); plt.show()


plot_screen(scr, f'{PLANE}: stage-1 screen (phantom v2)')
print(f"{'alpha_min':>9s} {'flagged':>8s} {'res/noise median':>17s} {'max':>6s}  flagged runs")
for j, rr in scr.items():
    q = np.array([r['final_res'] / r['noise_floor'] for r in rr])
    bad = [f"x={r['x']} e{r['e']}" for r in rr if r['final_res'] > S1.FLAG * r['noise_floor']]
    print(f'{j:9.2f} {len(bad):5d}/{len(rr)} {np.median(q):17.2f} {q.max():6.2f}  {bad}')
PASSING = [j for j, rr in scr.items() if S1.n_flagged(rr) == 0]
print('passing:', PASSING)
""")

md(r"""
## 5. Full coronal planes: per-slice convergence

Three full planes (the approved budget): α_min = 0.3 (the V2 baseline, now on phantom v2), the
**lowest value that passed** the screen, and the **next grid value below it**, which failed the
screen in a few slices — the case the adaptive fallback (section 8) is for, and the one that
shows what less regularisation buys. All 112 readout slices × 2 echo groups each; the full plane
is the real stability test (224 slice runs vs 26 in the screen).
""")
code(r"""
PLANE_J = S1.planes_to_run(PASSING)
print('full planes for alpha_min', PLANE_J)
rec = {}
for j in PLANE_J:
    rec[j] = [mp.moba_recon(scans, e, PLANE, S1.opts_j(j), PHANTOM_DIR, PROC_DIR,
                            progress=PROC_DIR / 'moba_progress.log')[0] for e in (0, 1)]
    for e, r in enumerate(rec[j]):
        q = r['final_res'] / r['noise_floor']
        print(f"alpha_min {j}: echo {e}: {float(r['runtime_s']) / 60:5.1f} min, res/noise median {np.median(q):.2f}, "
              f"max {q.max():.2f} (x={int(np.argmax(q))}), flagged {int((q > S1.FLAG).sum())} {np.flatnonzero(q > S1.FLAG).tolist()}")
F.slice_qc({f'α_min {j}': rec[j] for j in PLANE_J}, f'{PLANE}: per-slice convergence, full plane (phantom v2)'); plt.show()
""")

md(r"""
## 6. Phantom numbers per α_min

T1 = 1/mean(R1*) over the two echo groups, NaN outside the head label (≥ 3), evaluated with
`pu.evaluate_t1` (eroded tissue masks; each lesion: voxels holding ≥ half its peak partial-volume
fraction). **Bias vs ideal** (the resolution-limited, fully sampled, noise-free reference) is the
reconstruction's own error. Below the table: medians and robust SD (1.4826 × MAD) in WM and GM,
and the histograms per α_min.
""")
code(r"""
T1 = {j: 1000 * mp.t1_from([r['maps'] for r in rec[j]], head) for j in PLANE_J}          # ms
rows = {j: pu.evaluate_t1(T1[j] / 1000, truth) for j in PLANE_J}
llr_f = PHANTOM_DIR / f'llr_{PLANE}.npz'
if llr_f.exists():
    T1_llr = 1000 * np.load(llr_f)['T1_s']; rows['LLR'] = pu.evaluate_t1(T1_llr / 1000, truth)
cols = list(rows)
r0 = rows[cols[0]]
print(f"{'region':18s} {'n':>5s} {'true':>6s} {'ideal':>6s}  " + ''.join(f"{('α ' + str(c)) if c != 'LLR' else 'LLR':>17s}" for c in cols))
for i, r in enumerate(r0):
    if r['region'] == 'CSF':
        continue
    print(f"{r['region']:18s} {r['n']:5d} {r['true_ms']:6.1f} {r['ideal_ms']:6.1f}  "
          + ''.join(f"{rows[c][i]['est_ms']:7.1f} ({rows[c][i]['bias_vs_ideal_pct']:+5.1f}%)" for c in cols))
print('cells: mean T1 (ms) and bias vs ideal; CSF omitted (fails for every setting, see V2 report)')
masks = F.tissue_masks(truth)
for name in ('WM', 'GM'):
    m = masks[name]
    print(f'{name}: ideal median {np.median(truth["T1_ideal_ms"][m]):.1f} | ' + ' | '.join(
        f"α {j}: median {np.nanmedian(T1[j][m]):.1f}, robust SD {1.4826 * np.nanmedian(np.abs(T1[j][m] - np.nanmedian(T1[j][m]))):.1f}"
        for j in PLANE_J))
F.bias_bars({f'α_min {j}': rows[j] for j in PLANE_J}, f'{PLANE}: mean T1 bias vs ideal per α_min (phantom v2)'); plt.show()
F.hist_by_setting({f'α_min {j}': T1[j] for j in PLANE_J}, truth, f'{PLANE}: T1 histograms per α_min (eroded masks)'); plt.show()
""")

md(r"""
## 7. Images for visual assessment

Coronal in-plane slices (1.8 × 1.8 mm; rows = α_min, first row the ideal), at the mid slice and the
slices through the lesion centres; shared window. Then the same as differences from the ideal,
a zoom on the lesion region, and the lesion panel (zoomed central slice of each lesion, per-voxel
T1 inside each lesion mask).
""")
code(r"""
LZ = sorted(set(z for _, _, z in F.lesion_slices(truth)))
SL = [truth['label'].shape[2] // 2] + LZ
vols = {f'α_min {j}': T1[j] for j in PLANE_J}
F.sweep_grid(vols, truth, SL, SP, f'{PLANE}: T1 per α_min (phantom v2)'); plt.show()
F.sweep_grid(vols, truth, SL, SP, f'{PLANE}: T1 − ideal per α_min', diff=True); plt.show()
xs_ = [x for x, _, _ in F.lesion_slices(truth)]
crop = (slice(min(xs_) - 12, max(xs_) + 13), slice(8, truth['label'].shape[1] - 8))
F.sweep_grid(vols, truth, LZ, SP, f'{PLANE}: zoom on the lesion band (x = {crop[0].start}-{crop[0].stop - 1})', crop=crop); plt.show()
jl = min(PLANE_J)
sel = {f'α_min {j}': T1[j] for j in sorted({jl, 0.3})}
F.lesion_panel(sel, truth, SP, f'{PLANE}: lesions, lowest stable α_min vs 0.3'); plt.show()
""")

md(r"""
## 8. Adaptive fallback and `-R 3`

**Adaptive fallback:** reconstruct at a lower α_min and replace only the slices the residual QC
flags by the α_min = 0.3 slices. With full planes at both values this costs nothing extra: the
table shows, per lower α_min, how many slices would be replaced and the resulting numbers.
**`-R 3`:** faster decay of α towards α_min (more Newton steps at the floor) at the lowest passing
value, screen only.
""")
code(r"""
for j in [j for j in PLANE_J if j != 0.3]:
    q = np.maximum(*(r['final_res'] / r['noise_floor'] for r in rec[j]))      # worst of the two echo groups
    bad = np.flatnonzero(q > S1.FLAG)
    maps = [r['maps'].copy() for r in rec[j]]
    for e in (0, 1):
        maps[e][bad] = rec[0.3][e]['maps'][bad]
    T1_fb = 1000 * mp.t1_from(maps, head)
    rr = pu.evaluate_t1(T1_fb / 1000, truth)
    print(f'alpha_min {j}: {len(bad)} slices replaced {bad.tolist()}; bias vs ideal: '
          + ', '.join(f"{r['region'].split(' (')[0]} {r['bias_vs_ideal_pct']:+.1f}%" for r in rr if r['region'] != 'CSF'))
if PASSING:
    jl = min(PASSING)
    r3 = S1.screen(jl, SCALING, extra=' -R 3', tag=f'j{jl} R3')
    q = np.array([r['final_res'] / r['noise_floor'] for r in r3])
    print(f'-j {jl} -R 3 screen: flagged {S1.n_flagged(r3)}/{len(r3)}, res/noise median {np.median(q):.2f}, max {q.max():.2f}')
    plot_screen({f'{jl}': scr[jl], f'{jl}, -R 3': r3}, f'{PLANE}: -R 2 (default) vs -R 3 at α_min {jl}')
""")

md(r"""
## 9. What these T1 maps are

Each `moba` run gives a **per-plane T1 map on that plane's own reconstruction grid**: here the
coronal plane, 112 × 100 × 44 voxels of 1.8 (S/I, readout) × 1.8 (L/R) × 5 mm (A/P). "In-plane"
(the coronal 1.8 × 1.8 mm view) is the native resolution; through-plane it is 5 mm. The map is
built slice-wise along the readout (S/I): each readout position is an independent 2D problem
(L/R × A/P), which is exact because the readout is fully sampled and Cartesian. Nothing here is
isotropic yet: combining the three planes into the 1.8 mm isotropic T1 map (3-plane super-
resolution) is a later step (`Recon_MOBA_V2.ipynb` section 17); the axial and sagittal planes
are not reconstructed in this stage.
""")

REPORT = Path(__file__).with_name('report_s1.md')
md(REPORT.read_text() if REPORT.exists() else '## 10. Decision\n\n*(written after the runs)*')

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}
out = sys.argv[1] if len(sys.argv) > 1 else 'Recon_MOBA_Tuning_S1.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
