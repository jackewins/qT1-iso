"""Builds Recon_MOBA_V2.ipynb (code + markdown) from Recon_MOBA_V1.ipynb. Run from the repo root.

Cells that V1 verified and V2 does not change (model check, TI stacking, per-TI phase, the
3-plane super-resolution discussion) are copied verbatim from V1; everything else is new.
The report cell (last) is written after the notebook has been executed (REPORT below).
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

V1 = json.load(open('Recon_MOBA_V1.ipynb'))
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip('\n')))


def v1(i):
    c = V1['cells'][i]
    (md if c['cell_type'] == 'markdown' else code)(''.join(c['source']))


def v1_md_replace(i, pairs):
    s = ''.join(V1['cells'][i]['source'])
    for a, b in pairs:
        assert a in s, (i, a[:60])
        s = s.replace(a, b)
    md(s)


# ============================================================================================
md(r"""
# Model-based T1 reconstruction (BART `moba -L`) — synthetic phantom, V2: first stable version

Model-based alternative to the joint-LLR reconstruction (`RECON_COMPARISON.md`, method 2):
`moba` estimates the parameter maps of the IR signal model — and the coil sensitivities —
**directly from the undersampled k-space of all four TIs at once** (Wang et al., MRM
2018;79:730, doi:10.1002/mrm.26726), with the iteratively regularised Gauss-Newton method
(IRGNM); each Newton step's linear subproblem is solved by FISTA with a joint l1-wavelet
penalty on the parameter maps.

**Data: synthetic phantom only** (`synth/phantom.py`, static head); no subject data are used.

**What V2 adds to V1** (`Recon_MOBA_V1.ipynb`, which stays as the record of the first build).
V1 diverged in many readout slices. Section 7 diagnoses why on single readout slices and
arrives at a configuration that converges in every slice; the rest of the pipeline is V1's,
now executed end to end (V1's sections 10-15 were never run).

| | V1 | V2 | why (section 7) |
|---|---|---|---|
| data scaling | `--normalize_scaling --scale_data 5000 --scale_psf 1000`: every slice's data norm set to 5000 | **fixed per plane**: `--scale_data K/σ` (σ = k-space noise SD), `--scale_psf` = moba's own PSF normalisation | per-slice normalisation gives noisy (peripheral) slices the weakest regularisation relative to their noise |
| coil smoothness | `--sobolev_a 880` (default) | **`--sobolev_a 220`** | the default coil model cannot represent the coils (10-13 % error); the error ends up in the maps |
| α_min (`-j`) | 0.01 | **0.3** | below ~0.1 the late, weakly regularised Newton steps diverge in some slices |
| Newton steps (`-i`) | 8 | **10** | two more steps at α ≈ α_min (converged schedule) |

| step | what | section |
|---|---|---|
| 0 | verify that the protocol's IR equation is exactly `moba -L`'s model; TI file units/dims | 3 |
| 1 | load and stack the 4 TIs per plane and echo group (`pipeline_utils.stack_tis`) | 4 |
| 2 | remove the per-TI global phase offsets (moba shares one complex image across TIs) | 5 |
| 3 | coil sensitivities: what `moba -L` can and cannot take | 6 |
| 4 | **stability diagnosis on single readout slices** (new) | 7 |
| 5 | `moba -L` per echo group → M_ss, M0', R1*, per-slice convergence check | 8-9 |
| 6 | T1 = 1/R1*, combine echo groups, evaluate vs truth | 10-11 |
| 7 | model-consistency checks: synthesised TI images, k-space residual, M0'/M_ss | 12-13 |
| 8 | coil-map QC, LLR reference for context, all planes | 14-16 |
| 9 | what a per-plane T1 output means for 3-plane super-resolution; report | 17-18 |

**Not tuned yet.** V2 is the first configuration that is *stable*; α_min, iterations,
regulariser and coil initialisation are tuned next, stage by stage with the researcher
(section 18 lists the plan).
""")

md("## 1. Environment")
code(r"""
import os
import sys
import re
import time
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

os.environ.setdefault('BART_TOOLBOX_PATH', str(Path('~/bart').expanduser()))
REPO = Path.cwd()
sys.path.insert(0, str(REPO / 'synth'))
sys.path.insert(0, str(REPO / 'moba_dev'))
import pipeline_utils as pu                      # shared conventions (do not re-implement)
from pipeline_utils import bart
from phantom import load_scans
import diag_slice as ds                          # single-readout-slice moba harness (section 7)

%matplotlib inline
plt.rcParams['figure.facecolor'] = 'white'
BART_BIN = os.path.join(os.environ['BART_TOOLBOX_PATH'], 'bart')
print(subprocess.run([BART_BIN, 'version'], capture_output=True, text=True).stdout.strip(),
      '|', os.environ['BART_TOOLBOX_PATH'], '|', os.cpu_count(), 'CPUs')
""")

md(r"""
## 2. Configuration

The phantom lives outside the repo (`python synth/phantom.py ~/work/phantom`); derived
data are cached in `PROC_DIR` (also outside the repo). Delete a cache file to force a re-run.

`moba` options, V2 (**stable, not tuned**; section 7 derives each one):

| option | value | meaning |
|---|---|---|
| `-L` | | 3-parameter IR model S = M_ss − (M_ss + M0') e^(−TI·R1*) |
| `-l1` | | joint l1-wavelet penalty on the 3 parameter maps (moba default) |
| `-i` | 10 | IRGNM (Newton) steps |
| `-C` | 100 | max. FISTA iterations per Newton step (moba caps step n at min(C, 10·2^n)) |
| `-j` | 0.3 | minimum regularisation α_min; α_n = α_min + (1 − α_min)/2^n (`-R 2` default) |
| `--sobolev_a` | 220 | coil smoothness weight (moba default 880); `b` stays 32 |
| `--scale_data` | K/σ, K = 4.545 | data scale referenced to the k-space noise SD σ (= 45 in the coronal plane) |
| `--scale_psf` | 1000/‖PSF‖ | moba's own PSF normalisation (constant within a plane) |
| init | M_ss = M0' = 1, R1* = 1 s⁻¹ | moba default |
| R1* ≥ 0 | | moba default (last map constrained) |

`V1_OPTS` is kept for the section-7 comparison.
""")
code(r"""
PHANTOM_DIR = Path(os.environ.get('QT1_PHANTOM_DIR', '~/work/phantom')).expanduser()
PROC_DIR    = PHANTOM_DIR / 'moba'            # derived data, outside the repo
PROC_DIR.mkdir(exist_ok=True)

PLANE          = 'COR'                        # end-to-end validation plane
RUN_ALL_PLANES = os.environ.get('MOBA_ALL_PLANES', '0') == '1'   # section 16 (AX, SAG: ~45 min each on 4 cores)

# ---- moba, V2 (stable, untuned) -------------------------------------------------
MOBA_NEWTON     = 10
MOBA_INNER      = 100
MOBA_REG        = '-l1'
MOBA_ALPHA_MIN  = 0.3
MOBA_SOBOLEV_A  = 220
NOISE_REF_SCALE = 4.545       # --scale_data = NOISE_REF_SCALE / sigma_k  (45 for the coronal plane)
MOBA_OPTS = f'-L {MOBA_REG} -i {MOBA_NEWTON} -C {MOBA_INNER} -j {MOBA_ALPHA_MIN} --sobolev_a {MOBA_SOBOLEV_A}'
MOBA_JOBS = os.cpu_count()    # readout slices reconstructed in parallel, 1 thread each
V1_OPTS = '-L -l1 -i 8 -C 100 -j 0.01 --normalize_scaling --scale_data 5000 --scale_psf 1000'

# ---- per-TI global phase (section 5) -----------------------------------------
PHASE_CORR  = True       # demodulate the per-TI global phase before moba
PHASE_TAPER = 0.15       # Gaussian k-space taper (SD as a fraction of the matrix) for the estimate

# ---- echo groups (section 10) -----------------------------------------------
ECHO_COMBINE = 'mean_R1'   # average R1* of the two echo groups, T1 = 1 / mean

MOBA_TAG = (f'v2_L{MOBA_REG}_i{MOBA_NEWTON}_C{MOBA_INNER}_j{MOBA_ALPHA_MIN}_a{MOBA_SOBOLEV_A}'
            f'_k{NOISE_REF_SCALE}_ph{int(PHASE_CORR)}')
scans = load_scans(PHANTOM_DIR)
PLANES = tuple(scans)
print('phantom:', PHANTOM_DIR, '| planes', PLANES, '| cache', PROC_DIR)
print('moba', MOBA_OPTS, '+ per-plane --scale_data / --scale_psf')
""")

# ---- sections 3-5 verbatim from V1 ----------------------------------------------------------
for i in range(5, 15):
    v1(i)

# ---- section 6: coils -----------------------------------------------------------------------
v1_md_replace(15, [
    ("compares with moba's own maps (and with the true maps).",
     "compares with moba's own maps (and with the true maps) — section 14 here."),
    ("""**Open choice:** convert `shared_selfcal(nmaps=1)` maps into moba's k-space representation
and pass them via `ksp-sens` as the initial value (inverse Sobolev weighting is badly
conditioned, and they would still be updated), or accept moba's own estimate (V1).""",
     """**New in V2 — the coil model and `ksp-sens`.** moba's coils are c = fftmod(IFFT(w ⊙ ĉ)) on
its 2× oversampled grid, with Sobolev weights w = (1 + a·|k|²)^(−b/2) (`noir/model.c`,
`noir_calc_weights`); `moba_dev/diag_slice.py: coils_from_ksp` reproduces moba's own output
from its exported k-space coils (`--other export-ksp-sens`) to 2·10⁻⁶, and `ksp_from_coils`
inverts it (band-limited where w > 10⁻³ max w) to build a `ksp-sens` initial value from any
image-space maps. Two findings (section 7d):

* with the default `a = 880` the coil model **cannot represent the phantom's coils** better
  than 10-13 % (relative error inside the head); `a = 220` brings this to 4-6 %. The part of
  the coils the model cannot represent ends up in the parameter maps. V2 uses `a = 220`.
* initialising moba with the **true** coils (oracle, via `ksp-sens`) helps much less than
  the smaller `a`, so the coil *model*, not the coil *initialisation*, was limiting.
  Initialising from `shared_selfcal(nmaps=1)` is therefore left to the tuning stage.""")])

# ============================================================================================
md(r"""
## 7. Stability diagnosis on single readout slices (new in V2)

V1 diverged: the residual rose again in the late, weakly regularised Newton steps, and R1*
collapsed towards 0 in some voxels, mostly in readout slices towards the head periphery.
Because the readout is fully sampled and Cartesian, every readout position is an
independent 2D problem (section 8), so the solver can be studied one slice at a time
(35-50 s per slice, single thread). `moba_dev/diag_slice.py` runs one slice exactly as the
full reconstruction does and reports:

* **raw-data residual** ‖y − A(c·ρ)‖/‖y‖: moba's coils × model images put back on the data
  scale and taken to k-space with the NUFFT, next to the **noise floor** √N·σ/‖y‖.
  moba's image × coil product equals the true image × 2·s_data/s_psf (the 2: the unitary FFT
  on moba's 2× oversampled grid; confirmed on a converged slice, where the residual then
  equals the noise floor to 3 digits). A converged, regularised fit sits at or somewhat below
  the noise floor (~5 real unknowns per voxel against ~11 real samples); **far above it
  means the solution is broken**.
* **collapsed voxels**: fraction of GM/WM/lesion voxels with T1 = 1/R1* > 1 s;
* median T1 per tissue vs the resolution-limited ideal on that slice.

The **screen** uses 13 readout slices, x = 8, 16, …, 104 (bottom of the head to its top;
x = 8 and 104 hold only scalp/skull). All results are cached in `PROC_DIR/diag/cache`
(keyed by plane, slice, echo and options); a fresh run of this section takes ~25 min on 4
cores.
""")
code(r"""
XS = list(range(8, 112, 8))
FIXED = '--scale_data 45 --scale_psf 17.971312'      # coronal: noise-referenced data scale, moba's PSF scale
V2_SLICE = f'-L -l1 -C 100 -i 10 {FIXED} --sobolev_a 220 -j 0.3'     # = MOBA_OPTS + the coronal scaling
truth = dict(np.load(PHANTOM_DIR / f'truth_{PLANE}.npz'))


def screen(opts, e=0, xs=XS, tag=''):
    return ds.run_many([dict(plane=PLANE, x=x, opts=opts, e=e, tag=f'{tag} x={x}') for x in xs], verbose=False)


def bias(r, name):
    v = r['rows'].get(name)
    return 100 * (v['med'] - v['ideal']) / v['ideal'] if v else np.nan


def plot_screens(runs, title):
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
    for (lab, rr), col in zip(runs.items(), ('#c0504d', '#1f4e79', '#7fa7cf', '#9bbb59', '#8064a2')):
        x = [r['x'] for r in rr]
        ax[0].semilogy(x, [r['final_res'] / r['noise_floor'] for r in rr], 'o-', color=col, label=lab)
        ax[1].plot(x, [100 * r['rows']['collapsed_frac'] for r in rr], 'o-', color=col, label=lab)
        ax[2].plot(x, [bias(r, 'WM') for r in rr], 'o-', color=col, label=f'{lab}: WM')
        ax[2].plot(x, [bias(r, 'GM') for r in rr], 's--', color=col, alpha=0.6, label=f'{lab}: GM')
    ax[0].axhline(1, color='0.4', lw=0.8); ax[0].set_ylabel('raw residual / noise floor')
    ax[1].set_ylabel('collapsed brain voxels (%)'); ax[2].set_ylabel('median T1 vs ideal (%)')
    ax[2].set_ylim(-40, 40); ax[2].axhline(0, color='0.4', lw=0.8)
    for a in ax:
        a.set_xlabel('readout slice x'); a.grid(alpha=0.25)
    ax[0].legend(fontsize=8); ax[2].legend(fontsize=7, ncol=2)
    plt.suptitle(title); plt.tight_layout(); plt.show()


def table(rr):
    print(f"{'':24s} {'x':>4s} {'res/noise':>9s} {'coll%':>6s} {'WM med/ideal':>13s} {'GM med/ideal':>13s} {'s_data':>7s}")
    for r in rr:
        w, g = r['rows'].get('WM'), r['rows'].get('GM')
        f = lambda v: f"{v['med']:6.0f}/{v['ideal']:<6.0f}" if v else ' ' * 13
        print(f"{r['tag']:24s} {r['x']:4d} {r['final_res'] / r['noise_floor']:9.2f} "
              f"{100 * r['rows']['collapsed_frac']:6.1f} {f(w)} {f(g)} {r['scale_data']:7.1f}")
""")

md(r"""
### 7a. V1 reproduces the divergence

V1's options on the 13-slice screen (echo group 0):
""")
code(r"""
v1_screen = screen(V1_OPTS, tag='V1')
table(v1_screen)
plot_screens({'V1': v1_screen}, f'{PLANE}: V1 options, single readout slices, echo 0')
""")

md(r"""
In moba's own residual metric (V1's log: the residual *entering* each Newton step) the
divergence looked peripheral. The raw-data residual after the last step shows it is broader:
**10 of 13 slices are broken** (residual 3-90× the noise floor, 4-19 % collapsed brain voxels),
central ones included (x = 56, 64); only x = 72 and the scalp-only slices x = 8 and 104 fit the
data. The last column is the data scale `--normalize_scaling` applied to each slice (7c).
""")

md(r"""
### 7b. What goes wrong: the Newton steps of one slice

`moba` is deterministic (fixed random wavelet shifts, fixed α and FISTA schedules), so a run
with `-i k` is exactly the state after k Newton steps. Slice x = 56 (central) with V1's
options, k = 1 … 8: residual, the number of voxels where R1* sits at its bound 0 (by tissue
label: fat, bone, CSF, GM, WM, and the background), and the largest |M_ss|.
""")
code(r"""
V1_BASE = '-L -l1 -C 100 -j 0.01 --normalize_scaling --scale_data 5000 --scale_psf 1000'
trace = ds.run_many([dict(plane=PLANE, x=56, opts=f'{V1_BASE} -i {k}', e=0, tag=f'i={k}') for k in range(1, 9)],
                    verbose=False)
lab56 = truth['label'][56]
print(f"{'step':5s} {'alpha':>6s} {'FISTA':>5s} {'res/noise':>9s} {'R1*=0 voxels: fat bone CSF  GM  WM   bg':>40s} {'max|Mss|':>9s} {'where':>12s}")
for k, r in enumerate(trace, start=1):
    R1 = np.real(r['maps'][..., 2]); A = np.abs(r['maps'][..., 0]); at0 = R1 <= 1e-3
    i = np.unravel_index(np.argmax(A), A.shape)
    alpha = 0.01 + 0.99 / 2 ** (k - 1)
    print(f"{k:5d} {alpha:6.3f} {min(100, 10 * 2 ** (k - 1)):5d} {r['final_res'] / r['noise_floor']:9.2f}  "
          + ' '.join(f'{int((at0 & (lab56 == l)).sum()):4d}' for l in range(1, 6)) + f' {int((at0 & (lab56 == 0)).sum()):5d}'
          + f"  {A.max():9.1f} {'bg' if lab56[i] == 0 else 'label %d' % lab56[i]:>8s} {str(tuple(int(v) for v in i)):>9s}")

fig, axes = plt.subplots(3, 5, figsize=(18, 5.6))
for c_, k in enumerate((3, 4, 6, 7, 8)):
    m = trace[k - 1]['maps']
    for r_, (v, ttl, kw) in enumerate(((np.log10(np.abs(m[..., 0]) + 1e-3), 'log10|M_ss|', dict(vmin=-1, vmax=2)),
                                       (np.real(m[..., 2]), 'R1* (1/s)', dict(vmin=0, vmax=6, cmap='magma')),
                                       (np.real(m[..., 2]) <= 1e-3, 'R1* at bound 0', dict(cmap='gray')))):
        axes[r_, c_].imshow(v.T, origin='lower', **kw); axes[r_, c_].axis('off')
        axes[r_, c_].set_title(f'after step {k}: {ttl}', fontsize=9)
plt.suptitle(f'{PLANE} readout slice 56, V1 options (pe1 horizontal, pe2 vertical)'); plt.tight_layout(); plt.show()
""")

md(r"""
**Mechanism.** The residual falls until step 6-7 and then jumps above the data norm in the
last step, while R1* hits its bound 0 in hundreds of voxels of *every* tissue and of the
background (so this is not a long-T1/CSF effect), and the largest |M_ss| sits in the
background near the edge of the FOV. Two properties of the solver combine:

1. **A null space at the bound.** At R1* = 0 the model is S(TI) = −M0' for every TI, so
   ∂S/∂M_ss = 1 − e^(−TI·R1*) = 0: M_ss drops out of the data term and is held only by the
   (by then weak) wavelet penalty. It drifts; when a later update moves R1* off zero, the
   inflated M_ss appears in the forward model at once.
2. **No step control.** Each Newton step's linearised problem is solved for x_{n+1}
   directly (`irgnm2`), and with `-l1` the maps have no quadratic (proximal) term — only the
   coils do (α‖ĉ‖²). What limits a step is α_n and FISTA's iteration cap min(C, 10·2^n),
   i.e. early stopping. Late steps (α → α_min, 100 FISTA iterations) can therefore take
   large, poorly constrained steps. (This also explains two later observations: more FISTA
   iterations, `-C 200`, *destabilise* a borderline slice; and BART's damped update
   `--pusteps/--ratio` does not help, because the overshoot happens inside one linearised
   solve.)

The next three subsections are the three changes that remove it.
""")

md(r"""
### 7c. Data scaling: per-slice normalisation gives noisy slices the least regularisation

`--normalize_scaling --scale_data 5000` rescales **each slice's** gridded data to norm 5000.
The noise SD σ is the same in every slice, but the signal is not: peripheral slices hold a
small cross-section of the head, so noise is a large part of their norm (noise floor 0.39 at
x = 26 vs 0.16 at x = 76) and their scale factor is large (s_data ≈ 100 vs ≈ 41). In moba's
units their noise is then ~2.5× larger than in central slices, against the **same** α — the
least informative slices get the weakest effective regularisation.

With a noise-referenced, fixed scale the ratio of regulariser to noise is the same in every
slice: in moba's units the maps are (s_data/s_psf) × the true maps, so for the M-maps the
l1 term relative to the data term scales as α/s_data, and relative to the noise as
α/(s_data·σ). Keeping **s_data·σ constant** keeps that ratio fixed across slices and planes.
V2 uses `--scale_data K/σ` with K = 4.545 (= 45 in the coronal plane, the value V1's
normalisation gave central slices) and `--scale_psf` = moba's own PSF normalisation
(1000/‖PSF‖, identical for every slice of a plane). A side benefit: the maps of all slices are
on one intensity scale (V1 needed a per-slice `rel_scale`). This also removes the reason V1
gave for 3D failing (the fixed-norm scaling did not adapt to volume size); 3D is not retried
here (tuning plan, section 18).
""")
code(r"""
x_ = [r['x'] for r in v1_screen]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(x_, [r['scale_data'] for r in v1_screen], 'o-'); ax[0].axhline(45, color='#c0504d', ls='--', label='V2 (fixed): 45')
ax[0].set_ylabel('s_data applied by --normalize_scaling'); ax[0].legend()
ax[1].plot(x_, [r['noise_floor'] for r in v1_screen], 'o-'); ax[1].set_ylabel('noise floor √N·σ/‖y‖')
for a in ax:
    a.set_xlabel('readout slice x'); a.grid(alpha=0.25)
plt.tight_layout(); plt.show()

FIX = '-L -l1 -i 8 -C 100 -j 0.01 --scale_data 45 --scale_psf 17.971312'
cmp = []
for lab, o in (('V1 (normalised)', V1_OPTS), ('fixed scaling, j 0.01', FIX), ('fixed scaling, j 0.1', FIX.replace('-j 0.01', '-j 0.1'))):
    cmp += ds.run_many([dict(plane=PLANE, x=x, opts=o, e=0, tag=lab) for x in (26, 56, 93)], verbose=False)
table(cmp)
""")

md(r"""
Fixed scaling alone improves the bottom slice (x = 26: residual 23 → 1.35 × noise floor) but is
not sufficient; with α_min = 0.1 the bottom and central slices converge, the top slice (x = 93)
still diverges, and where it converges (α_min 0.2-0.3, 7f) its WM T1 is far too low.
""")

md(r"""
### 7d. The coil model: `--sobolev_a`

Where x = 93 converges (fixed scaling, α_min 0.2-0.3) WM comes out 15-30 % short. Three checks
show this is moba's coil estimation, not the data:

1. **Oracle:** per-TI SENSE (`pics`, l1-wavelet) with the *true* coil maps, PSIR and the
   grid fit gets WM and GM right in the same slices (table below).
2. **Intensity:** |M_ss| × coil RSS should be one constant × M0 × true RSS in every slice.
   With the default coil model it is ~14-15 in central slices but only 6-8 in the top slices
   — the image × coil product loses half of the brain signal there.
3. **Representation:** round-tripping the *true* coils through moba's coil model (section 6)
   leaves 10-13 % error with the default `a = 880`, 4-6 % with `a = 220`, ~2-3 % with 55.
""")
code(r"""
# (1) oracle: per-TI SENSE with the true coils + PSIR grid fit (pipeline_utils.fit_t1_grid)
Dx_, T2_, TI_f_, info_ = ds.prepare(PLANE, 0)
print('per-TI pics -R W:6:0:0.002 with the TRUE coils, PSIR, grid fit (median T1 / ideal, ms):')
for x in (26, 56, 76, 90, 93):
    sens_x = pu.unit_rss(truth['sens'][x][None])
    X = []
    for j in range(4):
        im, log, err = ds.bart_quiet(1, 'pics -e -S -R W:6:0:0.002 -i 100', np.ascontiguousarray(Dx_[:, x:x + 1, ..., j:j + 1]),
                                     sens_x, t=np.ascontiguousarray(T2_[..., j:j + 1]))
        X.append(np.squeeze(im))
    X = np.stack(X, -1)
    S = np.real(X * np.conj(X[..., -1:]) / np.maximum(np.abs(X[..., -1:]), 1e-9))
    T1o, _, _ = pu.fit_t1_grid(S[None], info_['TI'], info_['TR'], (truth['label'][x] >= 3)[None])
    lab = truth['label'][x]; ide = truth['T1_ideal_ms'][x]
    print(f'  x={x:3d}: ' + '  '.join(f"{n} {np.median(1000 * T1o[0][lab == l]):.0f}/{np.median(ide[lab == l]):.0f}"
                                     for l, n in ((5, 'WM'), (4, 'GM'))))

# (3) representation error of moba's coil model for the true coils
ny_, nz_ = info_['matrix'][1:]
crop2 = lambda a: a[0, ny_ - ny_ // 2: ny_ - ny_ // 2 + ny_, nz_ - nz_ // 2: nz_ - nz_ // 2 + nz_]
sob_default = ds.sobolev_weights
print('\nrelative error of the true coils after moba\'s coil model (inside the head):')
for a_ in (880., 220., 55.):
    ds.sobolev_weights = lambda n1, n2, a=a_: sob_default(n1, n2, a=a)
    errs = []
    for x in (26, 56, 76, 93):
        m = truth['sens'][x]; hd = truth['label'][x] >= 1
        back = crop2(ds.coils_from_ksp(ds.ksp_from_coils(m, 1e-3)))
        errs.append(np.linalg.norm((back - m)[hd]) / np.linalg.norm(m[hd]))
    print(f'  sobolev_a {a_:5.0f}: ' + '  '.join(f'x={x}: {v:.3f}' for x, v in zip((26, 56, 76, 93), errs)))
ds.sobolev_weights = sob_default
""")
code(r"""
# (2) + the effect on T1: default coil model vs a = 220 vs oracle coil initialisation (a = 880)
B = '-L -l1 -C 100 -j 0.3 -i 10 --scale_data 45 --scale_psf 17.971312'
xs5 = (26, 56, 76, 90, 93)
ks = {x: ds.ksp_from_coils(0.6 * truth['sens'][x], 1e-3) for x in xs5}   # 0.6: moba's typical coil RSS
runs = {}
runs['a = 880 (default)'] = ds.run_many([dict(plane=PLANE, x=x, opts='-L -l1 -C 100 --scale_data 45 --scale_psf 17.971312 -j 0.3 -i 10',
                                              tag='a 880') for x in xs5], verbose=False)
runs['a = 880, true-coil init'] = ds.run_many([dict(plane=PLANE, x=x, opts=B, tag='a 880 true init', ksp_sens=ks[x],
                                                    ksp_tag='true0.6_t1e-3') for x in xs5], verbose=False)
runs['a = 220'] = ds.run_many([dict(plane=PLANE, x=x, opts=B + ' --sobolev_a 220', tag='a 220') for x in xs5], verbose=False)


def intensity_ratio(r):
    x = r['x']; hd = truth['label'][x] >= 3
    rss = np.sqrt((np.abs(r['sens']) ** 2).sum(-1)); trss = np.sqrt((np.abs(truth['sens'][x]) ** 2).sum(-1))
    return np.median((np.abs(r['maps'][..., 0]) * rss / np.maximum(truth['M0'][x] * trss, 1e-6))[hd])


print(f"{'':26s} " + ''.join(f'{"x=%d" % x:>22s}' for x in xs5))
for lab, rr in runs.items():
    print(f'{lab:26s} ' + ''.join(f"{intensity_ratio(r):5.1f} {r['rows']['WM']['med']:4.0f}/{r['rows']['WM']['ideal']:<4.0f}"
                                  f"{r['final_res'] / r['noise_floor']:5.2f}".rjust(22) for r in rr))
print('columns: intensity ratio |M_ss|·RSS/(M0·RSS_true), WM median/ideal (ms), residual/noise')
""")

md(r"""
With `a = 220` the intensity ratio is uniform across slices (16-18), WM is within ~5 % of
ideal in every test slice (x = 93: 249 vs 262 ms; default: 194), GM improves, and the residual
reaches the noise floor. The oracle coil *initialisation* with the default coil model helps
much less — the limit was the representable coil space, not the starting point.
""")

md(r"""
### 7e. α_min: where the stability boundary is

With fixed scaling and `a = 220`, the 13-slice screen for both echo groups at α_min = 0.3 (V2)
and 0.1:
""")
code(r"""
v2_screen = {e: screen(V2_SLICE, e, tag=f'V2 e{e}') for e in (0, 1)}
j01_screen = {e: screen(V2_SLICE.replace('-j 0.3', '-j 0.1'), e, tag=f'j0.1 e{e}') for e in (0, 1)}
plot_screens({'V1, echo 0': v1_screen, 'V2 (α_min 0.3), echo 0': v2_screen[0], 'V2, echo 1': v2_screen[1],
              'α_min 0.1, echo 0': j01_screen[0], 'α_min 0.1, echo 1': j01_screen[1]},
             f'{PLANE}: single readout slices — V1 vs V2 vs α_min 0.1')
for lab, s in (('V2', v2_screen), ('alpha_min 0.1', j01_screen)):
    rr = s[0] + s[1]
    bad = [f"x={r['x']} e{r['e']}" for r in rr if r['final_res'] > 1.5 * r['noise_floor']]
    print(f"{lab:14s}: diverged (residual > 1.5 x noise floor) in {len(bad)}/{len(rr)} slice runs {bad}")
""")

md(r"""
At α_min = 0.3 every slice of both echo groups converges (residual 0.85-0.9 × noise floor,
no collapsed voxels, WM within ~3 % of ideal; the scalp-only slice x = 8 sits ~10 % above its
floor). At α_min = 0.1 the maps are sharper where it converges (GM closer to ideal), but
6 of 26 slice runs diverge (x = 48, 56, 64, 96). **V2 takes 0.3**, leaving margin; locating the boundary between 0.1
and 0.3 (or making α_min adaptive per slice) is the first tuning stage.
""")

md(r"""
### 7f. Levers that did not help (or did not transfer)

For completeness, the other candidate levers from the V1 report, on the bottom (26), central
(56) and top (93) test slices, echo 0. Each row changes one thing relative to the
configuration named in its label.
""")
code(r"""
F = '--scale_data 45 --scale_psf 17.971312'
LEVERS = [
    ('V1', V1_OPTS),
    ('V1 + lower bound -B 0.3', V1_OPTS + ' -B 0.3'),
    ('V1 + l2 on M_ss', V1_OPTS + ' --l2-on-parameters 1'),
    ('V1 + j 0.1', V1_OPTS.replace('-j 0.01', '-j 0.1')),
    ('fixed, j 0.1 + auto-norm -N', f'-L -l1 -C 100 -j 0.1 {F} -i 8 -N'),
    ('fixed, j 0.2', f'-L -l1 -C 100 -j 0.2 {F} -i 8'),
    ('fixed, j 0.2 + init R1* 3', f'-L -l1 -C 100 -j 0.2 {F} -i 8 --other pinit=1:1:3'),
    ('fixed, j 0.2 + -C 200', f'-L -l1 -C 200 -j 0.2 {F} -i 8'),
    ('fixed, j 0.2 i10 + pusteps r 0.5', f'-L -l1 -C 100 {F} -j 0.2 -i 10 --pusteps 10 --ratio 0.5'),
    ('fixed, j 0.3 i10 (a 880)', f'-L -l1 -C 100 {F} -j 0.3 -i 10'),
    ('fixed, j 0.3 i10, a 220 (V2)', f'-L -l1 -C 100 -j 0.3 -i 10 {F} --sobolev_a 220'),
]
lev = {lab: ds.run_many([dict(plane=PLANE, x=x, opts=o, tag=lab) for x in (26, 56, 93)], verbose=False) for lab, o in LEVERS}
print(f"{'':34s}" + ''.join(f"{'x=%d: res/noise coll%%  WM   GM' % x:>34s}" for x in (26, 56, 93)))
for lab, rr in lev.items():
    print(f'{lab:34s}' + ''.join(f"{r['final_res'] / r['noise_floor']:9.2f} {100 * r['rows']['collapsed_frac']:5.1f} "
                                 f"{r['rows']['WM']['med']:5.0f} {r['rows']['GM']['med']:5.0f}".rjust(34) for r in rr))
print('ideal (ms):' + ' ' * 23 + ''.join(f"{truth_ideal:>34s}" for truth_ideal in
      [f"WM {lev['V1'][i]['rows']['WM']['ideal']:.0f} GM {lev['V1'][i]['rows']['GM']['ideal']:.0f}" for i in range(3)]))
""")

md(r"""
* **Lower bound `-B`** (R1* ≥ 0.3 s⁻¹): voxels drift to the bound instead of to 0; no help.
* **l2 on M_ss** (`--l2-on-parameters 1`): damps the null space but biases the curve and
  destabilises the top slice.
* **`-N`** (per-map normalisation before thresholding): worse.
* **Initial R1* = 3 s⁻¹** (`--other pinit=1:1:3`): speeds up WM convergence at the top of
  the head (default init overshoots to T1 ≈ 70 ms after step 2 there and creeps back) but
  biases the long-T1 lesion low and does not survive more iterations; not needed with `a = 220`.
* **`-C 200`, damped updates (`--pusteps/--ratio`)**: more inner iterations destabilise a
  borderline slice; damping does not help (7b).
* **`--other pscale`** is *not* a valid lever for `-L` in BART v1.0.00: the T1 model is
  built with fixed scalings (`T1_create(..., conf->scaling_M0, conf->scaling_R1s, ...)` = 1),
  and `pscale` only divides the initial value and multiplies the output — the returned R1*
  would simply be wrong by that factor. The equivalent lever is the unit of the TI file
  (TI × k makes moba estimate R1*/k); it was not needed, because M_ss ≈ 10 and R1* ≈ 4 s⁻¹
  already give Jacobian columns of similar size.
* **Step size `-s`**: not tested separately; it only changes FISTA's step relative to 1/L
  (default 0.9), i.e. acts like the iteration cap.
""")

md(r"""
### 7g. V2 configuration

`-L -l1 -i 10 -C 100 -j 0.3 --sobolev_a 220 --scale_data K/σ --scale_psf 1000/‖PSF‖`
(K = 4.545). Slice-wise along the readout as in V1. The price of stability is
regularisation: α_min = 0.3 smooths more than V1 intended, which is expected to show as
partial-volume-like bias at small lesions and GM (section 11) — the subject of tuning.
""")

# ============================================================================================
md(r"""
## 8. `moba -L` reconstruction per echo group (full plane)

All four TIs of one echo group go into one `moba -L` problem: phase-demodulated k-space,
TI file (seconds), trajectory as wrapper keyword `t=T`, no `-p`.

**Slice-wise along the readout (as in V1).** The readout is fully sampled and Cartesian
(`traj[0]` = integer positions −n/2 … n/2−1), so an inverse FFT along the readout decouples
the 3D problem *exactly* into independent 2D problems, one per readout position, each with the
(pe1, pe2) trajectory of the scan. Each 2D problem is solved by `moba`
(`--img_dims 1:ny:nz`), and the slices are stacked. Regularisation and coil smoothness act
within each (pe1, pe2) plane only.

**Scaling (V2).** `--scale_psf` is moba's own PSF normalisation (1000/‖PSF‖; read from a
one-step moba run on the central slice, identical for all slices of a plane) and
`--scale_data` = K/σ with σ the plane's k-space noise SD (`info['noise_sd']`; on real data it
must be estimated, e.g. from the noise scan or the k-space periphery). moba's image × coil
product is then the true image × 2·s_data/s_psf in every slice (`rel_scale` = its inverse).

With a trajectory `moba` works on a **2× oversampled grid** and restricts the maps to the
central half (`-f 0.5` default for non-Cartesian): maps and coils are cropped to the central
(ny, nz). **Per-slice convergence check (new):** after each slice, the raw-data residual and
its noise floor (section 7) are stored; a slice whose residual exceeds 1.5× its noise floor
is flagged as diverged.
""")
code(r"""
def ti_file(scans_plane):
    return np.array([s['info']['TI_s'] for s in scans_plane], np.complex64).reshape(1, 1, 1, 1, 1, -1)


def plane_scaling(Dx, T2, TI_f, matrix, noise_sd):
    # s_psf: moba's own PSF normalisation (1000/||PSF||), read from a one-step run on the central
    # slice (the PSF is the same for every slice); s_data: noise-referenced, NOISE_REF_SCALE / sigma
    nx, ny, nz = matrix
    _, log, err = ds.bart_quiet(2, f'moba -L -i 1 -C 5 --normalize_scaling --scale_data 5000 --scale_psf 1000 '
                                   f'--img_dims 1:{ny}:{nz}', np.ascontiguousarray(Dx[:, nx // 2:nx // 2 + 1]), TI_f, t=T2)
    s_psf = float(re.findall(r'Scaling_psf: ([0-9.eE+-]+)', log)[-1])
    return NOISE_REF_SCALE / noise_sd, s_psf


def moba_slice(Dx, T2, TI_f, ny, nz, cmd, info_slice):
    res, log, err = ds.bart_quiet(2, cmd, Dx, TI_f, t=T2)
    if err or res is None:
        raise RuntimeError(f'moba failed ({err}):\n{log[-2000:]}')
    x, s = [np.squeeze(a) for a in res]                              # (2ny, 2nz, 3), (2ny, 2nz, coil)
    crop = lambda a: a[ny - ny // 2: ny - ny // 2 + ny, nz - nz // 2: nz - nz // 2 + nz]
    num = lambda key: float(re.findall(rf'{key}: ([0-9.eE+-]+)', log)[-1])
    r = dict(maps=crop(x).astype(np.complex64), sens=crop(s).astype(np.complex64),
             scale_data=num('Scaling'), scale_psf=num('Scaling_psf'),
             steps=np.array([float(v) for v in re.findall(r'Step: \d+, Res: ([0-9.eE+-]+)', log)]))
    r['final_res'], r['noise_floor'] = ds.data_residual(r, Dx, T2, info_slice)
    with open(PROC_DIR / 'moba_progress.log', 'a') as fh:          # progress of long runs (one line per slice)
        fh.write(f"{time.strftime('%H:%M:%S')} {info_slice['tag']} res/noise {r['final_res'] / r['noise_floor']:.2f}\n")
    return r


def moba_recon(scans_plane, e, plane):
    f = PROC_DIR / f'moba_{plane}_e{e}_{MOBA_TAG}.npz'
    if f.exists():
        return dict(np.load(f)), 'cached'
    info = scans_plane[pu.REF_TI_IDX]['info']
    matrix = info['matrix']
    nx, ny, nz = matrix
    T, D, P = pu.stack_tis(scans_plane, e)                   # P not passed: see section 4
    dphi = ti_phase_offsets(T, D, matrix) if PHASE_CORR else np.zeros(D.shape[5])
    D = (D * np.exp(-1j * dphi).reshape(1, 1, 1, 1, 1, -1)).astype(np.complex64)
    Dx = bart(1, 'fft -i -u 2', D)                           # readout k -> x (dim 1)
    if Dx is None or np.shape(Dx) != D.shape:
        raise RuntimeError('readout iFFT failed')
    T2 = T[:, :1].copy(); T2[0] = 0                          # (0, k_pe1, k_pe2) per sample
    TI_f = ti_file(scans_plane)
    s_data, s_psf = plane_scaling(Dx, T2, TI_f, matrix, info['noise_sd'])
    cmd = f'moba {MOBA_OPTS} --scale_data {s_data:.4f} --scale_psf {s_psf:.6f} -d 4 --img_dims 1:{ny}:{nz}'
    info_slice = dict(matrix=matrix, noise_sd=info['noise_sd'], TI=np.real(TI_f).ravel())
    t0 = time.time()
    with ThreadPoolExecutor(MOBA_JOBS) as ex:
        out = list(ex.map(lambda x: moba_slice(np.ascontiguousarray(Dx[:, x:x + 1]), T2, TI_f, ny, nz, cmd,
                                               dict(info_slice, tag=f'{plane} e{e} x={x}')), range(nx)))
    dt = time.time() - t0
    res = {k: np.stack([o[k] for o in out]) for k in ('maps', 'sens', 'scale_data', 'scale_psf', 'steps',
                                                      'final_res', 'noise_floor')}
    res.update(dphi=dphi, runtime_s=dt, cmd=cmd)
    res['rel_scale'] = 0.5 * res['scale_psf'] / res['scale_data']   # moba units -> data units
    np.savez(f, **res)
    return res, 'reconstructed'


rec = {}
for e in (0, 1):
    rec[PLANE, e], state = moba_recon(scans[PLANE], e, PLANE)
    r = rec[PLANE, e]
    q = r['final_res'] / r['noise_floor']
    print(f"{PLANE} echo {e}: {state}, {r['maps'].shape[0]} readout slices, "
          f"wall time {float(r['runtime_s']) / 60:.1f} min ({MOBA_JOBS} parallel jobs)\n  {r['cmd']}")
    print(f'  raw residual / noise floor per slice: median {np.median(q):.2f}, range {q.min():.2f}-{q.max():.2f}; '
          f'diverged (> 1.5): {int((q > 1.5).sum())} slices {np.flatnonzero(q > 1.5).tolist()}')
""")

md(r"""
## 9. Convergence per slice and parameter maps

Top: the raw-data residual relative to the noise floor for every readout slice (both echo
groups) — the full-plane version of the section-7 screen. Then the maps, mid pe2 slice (the
1.8 × 1.8 mm in-plane view). With the fixed scaling, M_ss and M0' of all slices are on one
scale (moba units); their magnitude still carries the arbitrary image/coil split of the joint
estimation (the arcs in |M_ss| are the coil-map structure it absorbed; T1 is unaffected, R1*
being scale-free). Expected: R1* ≈ 3.8 s⁻¹ in WM, ≈ 3.2 in GM, ≈ 0.3 in CSF.
""")
code(r"""
info = scans[PLANE][pu.REF_TI_IDX]['info']
sp = pu.spacing_of(info); ASP = sp[0] / sp[1]
head = truth['label'] >= 3                                     # CSF, GM, WM, lesions
z0 = info['matrix'][2] // 2

fig, ax = plt.subplots(figsize=(10, 3))
for e, col in ((0, '#1f4e79'), (1, '#7fa7cf')):
    r = rec[PLANE, e]
    ax.plot(r['final_res'] / r['noise_floor'], 'o-', ms=3, color=col, label=f'echo group {e}')
ax.axhline(1, color='0.4', lw=0.8); ax.axhline(1.5, color='#c0504d', lw=0.8, ls='--', label='divergence flag')
ax.set_xlabel('readout slice x'); ax.set_ylabel('raw residual / noise floor'); ax.legend(fontsize=8)
ax.grid(alpha=0.25); ax.set_title(f'{PLANE}: per-slice convergence, V2'); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
for r_, e in enumerate((0, 1)):
    m = rec[PLANE, e]['maps']
    R1 = np.real(m[..., 2])
    vmax = np.percentile(np.abs(m[..., 0])[head], 99.5)
    panels = [(np.abs(m[..., 0]), '|M_ss|', dict(cmap='gray', vmin=0, vmax=vmax)),
              (np.abs(m[..., 1]), "|M0'|", dict(cmap='gray', vmin=0, vmax=vmax)),
              (R1, 'R1* (1/s)', dict(cmap='magma', vmin=0, vmax=6)),
              (1000 / np.maximum(R1, 1e-3), 'T1 = 1/R1* (ms)', dict(cmap='viridis', vmin=150, vmax=420))]
    for c_, (v, ttl, kw) in enumerate(panels):
        ax = axes[r_, c_]
        im = ax.imshow(v[:, :, z0], aspect=ASP, **kw)
        ax.set_title(f'echo group {e}: {ttl}', fontsize=10); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(f'{PLANE}: moba -L parameter maps (V2), slice {z0}'); plt.tight_layout(); plt.show()
""")

v1_md_replace(20, [("## 9. T1 and echo-group combination", "## 10. T1 and echo-group combination")])
v1(21)

md(r"""
## 11. Validation against the truth

`pu.evaluate_t1`: eroded tissue masks and per-lesion masks; **bias vs ideal** (the
resolution-limited, noise-free, fully sampled reference) is the reconstruction's own error
and the number to compare methods on. Means are reported by `evaluate_t1`; with V2 they are no
longer dominated by collapsed voxels (V1: WM mean +134 %, median −3.6 %).
""")
code(r"""
rows = pu.evaluate_t1(T1_moba, truth)
pu.print_eval(rows, f'\n{PLANE}: moba -L V2, echo groups combined ({ECHO_COMBINE}), {MOBA_TAG}')
rows_e = {}
for e in (0, 1):
    rows_e[e] = pu.evaluate_t1(T1_e[e], truth)
    pu.print_eval(rows_e[e], f'\n{PLANE}: moba -L V2, echo group {e} only')
from scipy.ndimage import binary_erosion
for l, n in ((5, 'WM'), (4, 'GM')):
    m = binary_erosion(truth['label'] == l) & np.isfinite(T1_moba)
    v = 1000 * T1_moba[m]
    print(f'{n}: median {np.median(v):.1f} ms (ideal median {np.median(truth["T1_ideal_ms"][m]):.1f}), '
          f'IQR {np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f}, T1 > 1 s in {100 * np.mean(v > 1000):.2f} % of voxels')
""")
code(r"""
T1_true = truth['T1_ms']; T1_ideal = truth['T1_ideal_ms']
lesion_z = [int(np.argmax(fr.sum(axis=(0, 1)))) for fr in truth['lesion_frac_by_id']]
slices = sorted(set([z0] + lesion_z))[:3]
fig, axes = plt.subplots(len(slices), 4, figsize=(17, 4.2 * len(slices)))
axes = np.atleast_2d(axes)
for r_, z in enumerate(slices):
    for c_, (v, ttl, kw) in enumerate([
            (1000 * T1_moba[:, :, z], 'moba T1 (ms)', dict(cmap='viridis', vmin=150, vmax=420)),
            (T1_ideal[:, :, z], 'ideal (resolution-limited) T1 (ms)', dict(cmap='viridis', vmin=150, vmax=420)),
            (1000 * T1_moba[:, :, z] - T1_ideal[:, :, z], 'moba − ideal (ms)', dict(cmap='RdBu_r', vmin=-60, vmax=60)),
            (1000 * T1_moba[:, :, z] - T1_true[:, :, z], 'moba − true (ms)', dict(cmap='RdBu_r', vmin=-60, vmax=60))]):
        ax = axes[r_, c_]
        im = ax.imshow(np.where(head[:, :, z], v, np.nan), aspect=ASP, **kw)
        ax.contour(truth['lesion_fraction'][:, :, z], [0.25], colors='k', linewidths=0.6)
        ax.set_title(f'slice {z}: {ttl}', fontsize=10); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(f'{PLANE}: moba T1 vs truth (black contours: lesions)'); plt.tight_layout(); plt.show()
""")
v1(25)

v1_md_replace(26, [("## 11. Model-generated TI images and k-space residual",
                    "## 12. Model-generated TI images and k-space residual"),
                   ("after fitting **one** global\ncomplex scale per echo group (moba works on internally normalised data).",
                    "after fitting **one** global\ncomplex scale per echo group. With V2's known scaling (`rel_scale`) that\nfitted scale should be 1 — a check of the scaling bookkeeping.")])
code(''.join(V1['cells'][27]['source']).replace(
    "print('relative k-space residual ‖y − a·A(x)‖/‖y‖ per TI, (noise floor √N·σ/‖y‖)')",
    "print('relative k-space residual ‖y − a·A(x)‖/‖y‖ per TI, (noise floor √N·σ/‖y‖)')\n"
    "print('fitted global scale a (should be ≈ 1):', {e: np.round(res[e][2], 4) for e in (0, 1)})"))
v1(28)
v1_md_replace(29, [("## 12. Is the free M0' consistent", "## 13. Is the free M0' consistent")])
v1(30)
v1_md_replace(31, [("## 13. Coil-map QC", "## 14. Coil-map QC")])
v1(32)
v1_md_replace(33, [("## 14. LLR reference", "## 15. LLR reference")])
v1(34)
v1_md_replace(35, [("## 15. All three planes", "## 16. All three planes"),
                   ("(`RUN_ALL_PLANES`)", "(`RUN_ALL_PLANES`, set by the environment variable `MOBA_ALL_PLANES=1`)")])
code(''.join(V1['cells'][36]['source']).replace(
    """            print(f"{p} echo {e}: {state}, moba runtime {float(rec[p, e]['runtime_s']) / 60:.1f} min")""",
    """            q = rec[p, e]['final_res'] / rec[p, e]['noise_floor']
            print(f"{p} echo {e}: {state}, moba runtime {float(rec[p, e]['runtime_s']) / 60:.1f} min, "
                  f"residual/noise median {np.median(q):.2f}, max {q.max():.2f}, diverged {int((q > 1.5).sum())}")"""))
v1(37)
v1_md_replace(38, [("## 16. What a per-plane T1 output means", "## 17. What a per-plane T1 output means"),
                   ("(section 11)", "(section 12)")])

REPORT = Path(__file__).with_name('report_v2.md')
if REPORT.exists():
    md(REPORT.read_text().replace('TUNING_PLAN', Path(__file__).with_name('tuning_plan_v2.md').read_text().strip()))
else:
    md('## 18. Report (V2)\n\n*(written after execution)*')

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}}
out = sys.argv[1] if len(sys.argv) > 1 else 'Recon_MOBA_V2.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
