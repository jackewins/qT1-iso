"""Builds CALIPR_LesionBias_V1.ipynb (run from the repo root, then execute with nbconvert).
Cells are kept here so the notebook can be regenerated after edits; the executed notebook is
what gets committed."""
import sys
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md(r"""# Why is lesion T1 pulled towards white matter? — controlled experiments (CALIPR, synthetic phantom)

**Question (task 1).** On the static phantom both the LLR reference and CALIPR V1 bias lesion
T1 towards the surrounding white matter **beyond partial volume** (short-T1 lesions roughly
+9 to +15 % vs `ideal`, long-T1 lesions −6 to −8 %), and LLR's block size hardly changes it,
so part of the cause is probably shared. This notebook separates the candidate causes with
controlled experiments on the coronal plane, run for **CALIPR V1 only** (LLR is developed
locally; its numbers from `RECON_COMPARISON.md` are context only).

**Candidate causes and the experiment that isolates each**

| cause | how it could pull lesions towards WM | isolated by |
|---|---|---|
| pipeline (phases, PSIR combination, echo average, grid fit, evaluation) | systematic error outside the reconstruction | fully sampled, noise-free, true-map, unregularised control (section 4) |
| noise | noise-driven fit bias; the regulariser behaves differently at low SNR | noisy vs noise-free phantom (identical seed and phases) |
| 2 soft-SENSE map sets | twice the unknowns (under-determined at R ≈ 6 with 8 coils); set 1 can absorb signal | ENLIVE 2 vs 1 map set |
| coil-map estimation | ENLIVE maps estimated from the same undersampled data | ENLIVE 1 set vs the phantom's true maps |
| regularisation | wavelet shrinkage of the small, T1-carrying coefficients $c_2, c_3$ smooths them towards the neighbourhood | default λ vs λ → 0 (converged CG), plus a λ dose-response |
| undersampling | too little encoding to resolve the lesion; **the same pattern at every TI**, so what is lost is lost identically at every TI | what remains at λ → 0; per-TI SENSE with λ → 0 as the method-neutral floor |

Each row of the experiment ladder removes one cause relative to the previous row, so the
difference between consecutive rows is that cause's contribution (a sequential decomposition:
interactions are attributed to the step where they disappear).

All data are the synthetic phantom (`synth/phantom.py`); nothing here uses real subject data.
Reconstructions are run by `synth/run_bias_experiments.py` (cached outside the repo), so this
notebook only loads, evaluates and plots; it recomputes a condition only if its cache is missing."""))

C.append(md("## 1. Environment"))
C.append(code(r"""import os, sys, json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BART_PATH = os.environ.get('BART_TOOLBOX_PATH', '/Users/jackewins/bart/')
os.environ['BART_TOOLBOX_PATH'] = BART_PATH
sys.path.append(os.path.join(BART_PATH, 'python'))
REPO_DIR = Path(os.environ.get('QT1_REPO_DIR', Path.cwd()))
sys.path.insert(0, str(REPO_DIR / 'synth'))
import pipeline_utils as pu
import calipr as cal
import bias_diagnostics as bd
import run_bias_experiments as rbe
import phantom as ph

%matplotlib inline
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 72"""))

C.append(md(r"""## 2. Configuration

`PHANTOM_DIR` is the standard static phantom, `PHANTOM_NF_DIR` the same phantom generated with
`--noise 0` (same seed: the per-TI phase offsets and everything else are identical, only the
noise is absent). `CACHE` holds the runs of `run_bias_experiments.py`.

Conditions are named `<data>:<maps>:<method>:<reg>` (see the script's docstring). CALIPR is
V1 exactly (`pics -e -S -R W:7:0:0.005 -i 80 -U`, K = 3 dictionary basis, per-TI global phase
demodulation)."""))
C.append(code(r"""PHANTOM_DIR    = Path(os.environ.get('QT1_PHANTOM_DIR', Path.home() / 'work' / 'phantom'))
PHANTOM_NF_DIR = Path(os.environ.get('QT1_PHANTOM_NF_DIR', Path.home() / 'work' / 'phantom_nf'))
CACHE          = Path(os.environ.get('QT1_BIAS_CACHE', Path.home() / 'work' / 'bias'))
PLANE = 'COR'

LADDER = [  # (label, condition): each step removes one candidate cause
    ('0  noisy, ENLIVE 2 sets, λ=0.005', 'noisy:E2:CAL:def'),
    ('A1 noise-free, ENLIVE 2 sets',     'nf:E2:CAL:def'),
    ('A2 noise-free, ENLIVE 1 set',      'nf:E1:CAL:def'),
    ('A3 noise-free, true maps',         'nf:true:CAL:def'),
    ('A4 noise-free, true maps, λ→0',    'nf:true:CAL:zero300'),
]
EXTRA = [
    ('A4 λ→0, 100 CG iterations',        'nf:true:CAL:zero'),
    ('B1 true maps, λ=0.0005',           'nf:true:CAL:lam0.0005'),
    ('B1 true maps, λ=0.00005',          'nf:true:CAL:lam5e-05'),
    ('S  per-TI SENSE, λ→0 (no TI model)', 'nf:true:LLR:zero'),
]
os.environ['QT1_PHANTOM_DIR'], os.environ['QT1_PHANTOM_NF_DIR'] = str(PHANTOM_DIR), str(PHANTOM_NF_DIR)
scans = ph.load_scans(PHANTOM_DIR)[PLANE]
TI = np.array([s['info']['TI_s'] for s in scans]); TR = np.array([s['info']['TR_s'] for s in scans])
truth = dict(np.load(PHANTOM_DIR / f'truth_{PLANE}.npz'))
MASK = truth['label'] >= 3
sp = pu.spacing_of(scans[-1]['info']); ASP = sp[0] / sp[1]
print('TI (ms)', np.round(1000 * TI, 2), '| TR (s)', TR, '| voxel (mm)', np.round(sp, 3))"""))

C.append(md(r"""## 3. The sampling pattern is identical at every TI

The phantom uses the real protocol's sampling coordinates. Within a plane, all four TIs (and
both echo groups) sample **exactly the same** phase-encode positions (the only exception is
axial TI31). Sampling is variable density: local undersampling R ≈ 2–3 near the k-space centre
rising to ≈ 8–10 at the edge.

Why this matters for any TI-coupled model (LLR or subspace): with one pattern for all TIs, the
aliasing at a voxel is the same superposition of other voxels at every TI, so the aliased signal
has a perfectly valid IR curve. A temporal model cannot tell it from true signal; only coil
encoding and the spatial regulariser can. It also means that whatever k-space content the
reconstruction fails to recover is missing identically at every TI."""))
C.append(code(r"""spec = ph.load_spec(); by = {}
for s in spec['scans']:
    by.setdefault(s['plane'], []).append(s)
for p, sl in by.items():
    sl.sort(key=lambda s: s['TI_s']); m = sl[-1]['matrix']
    pts = [set(map(tuple, np.round(np.stack([c[0, :, 0] * m[1], c[1, :, 0] * m[2]], 1), 2)))
           for c in [np.load(ph.HERE / 'acq_spec' / s['coord_file']) for s in sl]]
    print(p, '  '.join(f"{s['name'].split('_')[1]}: {len(q & pts[-1])}/{len(q)} shared with TI800" for s, q in zip(sl, pts)))

c = np.load(ph.HERE / 'acq_spec' / [s for s in by[PLANE] if s['name'].endswith('TI800')][0]['coord_file'])
m = scans[-1]['info']['matrix']; k1, k2 = c[0, :, 0] * m[1], c[1, :, 0] * m[2]
r = np.sqrt((k1 / (m[1] / 2)) ** 2 + (k2 / (m[2] / 2)) ** 2)
g1, g2 = np.meshgrid(np.arange(-m[1] // 2, m[1] // 2), np.arange(-m[2] // 2, m[2] // 2), indexing='ij')
rg = np.sqrt((g1 / (m[1] / 2)) ** 2 + (g2 / (m[2] / 2)) ** 2)
edges = np.linspace(0, 1.4, 15); h, _ = np.histogram(r, edges); hg, _ = np.histogram(rg, edges)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(k1, k2, '.', ms=2); ax[0].set_xlabel('k pe1 (L/R)'); ax[0].set_ylabel('k pe2 (A/P, slice)')
ax[0].set_title(f'{PLANE}: sampled PE positions (all TIs, both echo groups)'); ax[0].set_aspect(1)
ax[1].plot(0.5 * (edges[1:] + edges[:-1]), hg / np.maximum(h, 1), 'o-')
ax[1].set_xlabel('normalised k radius'); ax[1].set_ylabel('local R (grid points / samples)')
ax[1].set_title('variable density'); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()"""))

C.append(md(r"""## 4. Evaluation checks before any reconstruction

**4a. The evaluation masks are off-centre (phantom truth issue, shared code — reported, not
changed here).** `truth['lesion_frac_by_id']` (like `lesion_fraction`, `T1_ms_pv` and `sens`)
is a block mean of the fine simulation grid, and those blocks are centred 0.25 voxel (in-plane)
and 0.375 voxel (through-plane, ≈ 1.9 mm) away from the reconstruction voxels. The masks used
by `evaluate_t1` therefore sit towards one edge of each lesion. The comparison stays fair
(ideal and estimate always use the same voxels), but the voxels include more partial volume,
which changes the magnitude of the numbers. Below, a correctly centred fraction is computed
(blocks centred on the reconstruction voxels) and every result is reported with **both**
masks.

**4b. Fully sampled control.** Same phantom, but every k-space point of the reconstruction
matrix sampled, noise-free, the true coil maps, no regularisation. Then SENSE has a closed form
(no solver, no iteration count): $x = \sum_c S_c^* \,\mathrm{trunc}(m\,e^{i\phi}S_c) / \sum_c |S_c|^2$.
Everything downstream (per-TI phase, `combine_maps`, echo average, `fit_t1_grid`,
`evaluate_t1`) is the real pipeline. A second variant truncates k-space at the pattern's actual
pe2 extent (|k₂| ≤ 20.92 instead of 22), to check whether the slightly smaller encoded extent
matters."""))
C.append(code(r"""f = CACHE / 'lesion_frac_centred_COR.npz'
if not f.exists():
    info = scans[-1]['info']; fac = ph.FACTOR
    X, Y, Z = ph.grid_coords(info['matrix'], info['fov_mm'], info['geom'], fac)
    spf = [fv / (n * k) for fv, n, k in zip(info['fov_mm'], info['matrix'], fac)]
    xyz = [X, Y, Z]
    for j, a in enumerate(info['geom']):
        xyz[ph.AXIS[a]] = xyz[ph.AXIS[a]] - (fac[j] - 1) / 2 * spf[j]       # centre each block
    _, _, _, les = ph.tissue_maps(*xyz, 0)
    np.savez(f, frac=np.stack([ph.block_mean((les == i + 1).astype(np.float32), fac) for i in range(len(ph.LESIONS))]))
truth_c = dict(truth, lesion_frac_by_id=np.load(f)['frac'])

def centroid(v):
    idx = np.indices(v.shape).reshape(3, -1); w = v.ravel(); return (idx * w).sum(1) / w.sum()
info = scans[-1]['info']
print(f"{'lesion':8s} {'analytic centre (vox)':>24s} {'stored-frac centroid':>24s} {'centred-frac centroid':>24s}  peak PV stored / centred")
for i, (l, (lr, ap, si), d) in enumerate(ph.LESIONS):
    xyz = {'X': si, 'Y': lr, 'Z': ap}
    cen = [xyz[a] / sp[j] + info['matrix'][j] // 2 for j, a in enumerate(info['geom'])]
    print(f"L{i+1} {d:2d}mm  {str(np.round(cen, 2)):>24s} {str(np.round(centroid(truth['lesion_frac_by_id'][i]), 2)):>24s} "
          f"{str(np.round(centroid(truth_c['lesion_frac_by_id'][i]), 2)):>24s}  {truth['lesion_frac_by_id'][i].max():.2f} / {truth_c['lesion_frac_by_id'][i].max():.2f}")"""))
C.append(code(r"""def fs_control(tag, kmax):
    f = CACHE / f'fs_control_{tag}_COR.npz'
    if not f.exists():
        r = bd.fully_sampled_control(PHANTOM_DIR, PLANE, kmax=kmax, verbose=False)
        S = np.mean([np.real(pu.combine_maps(x)) for x in r['img']], axis=0)
        T1, M0, R2 = pu.fit_t1_grid(S, TI, TR, MASK)
        np.savez(f, signed=S.astype(np.float32), T1_s=T1, R2=R2, img0=r['img'][0], img1=r['img'][1])
    return dict(np.load(f))

FS = {'FS': fs_control('FS', None), 'FS |k2|<=20.92': fs_control('FS_k2', (None, None, 20.921))}
ideal_rows = pu.evaluate_t1(truth['T1_ideal_ms'] / 1000, truth)
ideal_rows_c = pu.evaluate_t1(truth['T1_ideal_ms'] / 1000, truth_c)
rows_fs = {k: (pu.evaluate_t1(v['T1_s'], truth), pu.evaluate_t1(v['T1_s'], truth_c)) for k, v in FS.items()}
print(f"{'region':18s} {'true':>7s} | {'ideal':>7s} " + ' '.join(f'{k:>16s}' for k in FS) +
      f" | {'ideal (centred)':>15s} " + ' '.join(f'{k:>16s}' for k in FS))
for i, r in enumerate(ideal_rows):
    print(f"{r['region']:18s} {r['true_ms']:7.1f} | {r['ideal_ms']:7.1f} "
          + ' '.join(f"{rows_fs[k][0][i]['bias_vs_ideal_pct']:+15.1f}%" for k in FS)
          + f" | {ideal_rows_c[i].get('ideal_ms', np.nan):15.1f} "
          + ' '.join(f"{rows_fs[k][1][i].get('bias_vs_ideal_pct', np.nan):+15.1f}%" for k in FS))"""))
C.append(md(r"""**Reading.** The fully sampled control reproduces `ideal` to within ±0.3 % in every region
and lesion (either mask): the pipeline itself (phase handling, PSIR combination, echo averaging,
fit, evaluation, and the 3 %-offset true coil maps) adds no measurable bias. Cutting k-space to
the pattern's real pe2 extent moves lesions by 1–3 % with mixed sign, not systematically
towards WM. So any bias beyond these levels below is created by the **reconstruction of
undersampled data** (undersampling, regularisation, coil maps, map sets, noise).

The table also shows how much the mask offset matters for the *ideal* itself: e.g. the 4 mm
lesions read 293 / 225 ms with the stored mask and 316 / 204 ms with the centred one."""))

C.append(md(r"""## 5. The experiment ladder

Loads each condition from `CACHE` (running it with `run_bias_experiments.run` if missing: about
5 min for a CALIPR condition on 4 cores, 20 min for the λ → 0, 300-iteration one). The table is
bias vs `ideal` in %, with the stored masks (the numbers comparable to earlier tables) and the
centred masks. The same `ideal` is used for noisy and noise-free data (identical object)."""))
C.append(code(r"""res = {}
for label, cond in LADDER + EXTRA:
    f = CACHE / f"{PLANE}_{cond.replace(':', '_')}.npz"
    if not f.exists():
        rbe.run(CACHE, cond)
    z = dict(np.load(f))
    res[cond] = dict(label=label, z=z, rows=pu.evaluate_t1(z['T1_s'], truth), rows_c=pu.evaluate_t1(z['T1_s'], truth_c))

def table(conds, key='rows'):
    names = [r['region'] for r in ideal_rows]
    print(f"{'condition':38s} " + ' '.join(f'{n.replace("lesion ", "L").replace(" mm)", ")"):>9s}' for n in names))
    for c in conds:
        rr = res[c][key]
        print(f"{res[c]['label']:38s} " + ' '.join(f"{r.get('bias_vs_ideal_pct', np.nan):+8.1f}%" for r in rr)
              + f"   [{float(res[c]['z']['seconds']) / 60:.1f} min]")

print('bias vs ideal (%), STORED lesion masks  (L1-4 long T1 338 ms, L5-8 short T1 182 ms; L4/L8 are 1-2 voxels)')
table([c for _, c in LADDER + EXTRA])
print('\nbias vs ideal (%), CENTRED lesion masks')
table([c for _, c in LADDER + EXTRA], 'rows_c')
print('\nT1 SD (ms) within WM / GM:  ' + '   '.join(f"{res[c]['label'][:2]}: {res[c]['rows'][0]['sd_ms']:.1f} / {res[c]['rows'][1]['sd_ms']:.1f}" for _, c in LADDER + EXTRA))"""))

C.append(md(r"""**Decomposition.** Each bar is the change in bias vs ideal (percentage points) from one step
of the ladder to the next — i.e. the contribution of the cause removed at that step. Lesions 1–3
and 5–7 (the 2 mm lesions are single voxels and are left out). Positive = towards longer T1."""))
C.append(code(r"""steps = [('noise', 0, 1), ('2 → 1 map set', 1, 2), ('ENLIVE → true maps', 2, 3),
         ('regularisation (λ → 0)', 3, 4)]
les = [3, 4, 5, 7, 8, 9]           # row indices: L1-3, L5-7
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), sharey=True)
for ax, key, ttl in ((axes[0], 'rows', 'stored masks'), (axes[1], 'rows_c', 'centred masks')):
    b = np.array([[res[c][key][i]['bias_vs_ideal_pct'] for i in les] for _, c in LADDER])   # (5, 6)
    w = 0.16; xx = np.arange(len(les))
    for s, (name, a, bb) in enumerate(steps):
        ax.bar(xx + (s - 1.5) * w, b[a] - b[bb], w, label=f'{name}')
    ax.plot(xx, b[0], 'k_', ms=22, mew=2, label='total bias, baseline (0)')
    ax.plot(xx, b[-1], 'r_', ms=22, mew=2, label='remaining at λ → 0 (undersampling)')
    ax.set_xticks(xx); ax.set_xticklabels([ideal_rows[i]['region'] for i in les], rotation=20)
    ax.axhline(0, color='k', lw=0.8); ax.grid(alpha=0.3, axis='y'); ax.set_title(f'contribution of each cause ({ttl})')
axes[0].set_ylabel('Δ bias vs ideal (percentage points)'); axes[1].legend(fontsize=8, loc='best')
plt.tight_layout(); plt.show()"""))

C.append(md(r"""## 6. Where the bias sits: lesion zooms

Each lesion at the slice of its peak (centred) partial-volume fraction, ±10 voxels: `ideal`,
then T1 for every step of the ladder. White contour: centred evaluation mask."""))
C.append(code(r"""def lesion_zoom(conds, h=10):
    frc = truth_c['lesion_frac_by_id']; order = [0, 1, 2, 4, 5, 6]
    rows_ = [('ideal', truth['T1_ideal_ms'])] + [(res[c]['label'][:2].strip() + ' ' + res[c]['label'][3:], 1000 * res[c]['z']['T1_s']) for c in conds]
    fig, axes = plt.subplots(len(rows_), len(order), figsize=(2.3 * len(order), 2.25 * len(rows_)), squeeze=False)
    for j, i in enumerate(order):
        x0, y0, z0 = np.unravel_index(np.argmax(frc[i]), frc[i].shape)
        sl = (slice(max(x0 - h, 0), x0 + h + 1), slice(max(y0 - h, 0), y0 + h + 1), z0)
        for r, (name, vol) in enumerate(rows_):
            a = axes[r, j]
            a.imshow(vol[sl], cmap='viridis', vmin=150, vmax=400, aspect=ASP)
            a.contour(frc[i][sl] >= 0.5 * frc[i].max(), [0.5], colors='w', linewidths=0.6)
            a.set_xticks([]); a.set_yticks([])
            if j == 0: a.set_ylabel(name, fontsize=7)
            if r == 0: a.set_title(f'L{i+1} ({ph.LESIONS[i][2]} mm, T1 {pu.TRUE_LESION_T1[ph.LESIONS[i][0]]:.0f}), z={z0}', fontsize=8)
    plt.suptitle('T1 (ms, 150-400): ideal and the ladder'); plt.tight_layout(); plt.show()

lesion_zoom([c for _, c in LADDER] + ['nf:true:LLR:zero'])"""))

C.append(md(r"""## 7. Line profiles through the lesions

Profiles through the peak voxel of the 10 mm and 4 mm lesion pairs along the read (S/I), pe1
(L/R) and pe2 (A/P, 5 mm slice) directions. Top: T1. Bottom: the normalised signal
$S(TI150)/S(TI800)$, which carries most of the T1 contrast (WM ≈ −0.13, short-T1 lesion
≈ +0.14, long-T1 lesion ≈ −0.29 at full volume). A reconstruction that only loses resolution
would show wider, lower lesion bumps; one that distorts contrast would show a lower bump of
the same width."""))
C.append(code(r"""def profiles(i, conds, h=8):
    frc = truth_c['lesion_frac_by_id'][i]; x0, y0, z0 = np.unravel_index(np.argmax(frc), frc.shape)
    cuts = {'read (S/I)': (slice(x0 - h, x0 + h + 1), y0, z0), 'pe1 (L/R)': (x0, slice(y0 - h, y0 + h + 1), z0),
            'pe2 (A/P, slice)': (x0, y0, slice(max(z0 - 4, 0), z0 + 5))}
    src = [('ideal', truth['T1_ideal_ms'], truth['S_ideal'], 'k', 2.2)] + \
          [(res[c]['label'][:2].strip(), 1000 * res[c]['z']['T1_s'], res[c]['z']['signed'], None, 1.2) for c in conds]
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.6))
    for j, (name, cut) in enumerate(cuts.items()):
        for lab, t1, S, col, lw in src:
            ratio = S[..., 1] / np.where(np.abs(S[..., -1]) > 0, S[..., -1], np.nan)
            axes[0, j].plot(t1[cut], color=col, lw=lw, label=lab); axes[1, j].plot(ratio[cut], color=col, lw=lw, label=lab)
        axes[0, j].set_title(f'L{i+1} ({ph.LESIONS[i][2]} mm): T1 along {name}', fontsize=9)
        axes[1, j].set_title('S(TI150)/S(TI800)', fontsize=9)
        for a in axes[:, j]: a.grid(alpha=0.3); a.set_xlabel('voxel')
    axes[0, 0].set_ylabel('T1 (ms)'); axes[0, 2].legend(fontsize=7)
    plt.tight_layout(); plt.show()

for i in (4, 0, 6, 2):          # L5, L1 (10 mm), L7, L3 (4 mm)
    profiles(i, [c for _, c in LADDER])"""))

C.append(md(r"""## 8. Is it extra partial volume? Curve-mixing fit

For each lesion, the mean normalised TI curve of the evaluation voxels (centred mask) in the
reconstruction is fitted as a mix of the ideal lesion curve and the ideal curve of the WM ring
around it: $s_{rec} = a\, s_{ideal,lesion} + (1-a)\, s_{ideal,WM}$. $a = 1$: nothing lost;
$a < 1$ with a **small residual**: the lesion is diluted with surrounding WM — exactly what extra
partial volume (a wider point-spread function) does, and a mechanism that is independent of TI.
A large residual means the curve is distorted in a TI-dependent way that mixing cannot
explain (e.g. contrast lost at some TIs only)."""))
C.append(code(r"""print(f"{'condition':38s} " + ' '.join(f'{"L"+str(i+1)+" a / res":>14s}' for i in (0, 1, 2, 4, 5, 6)))
mix = {}
for label, cond in [('FS control', None)] + LADDER + EXTRA:
    S = FS['FS']['signed'] if cond is None else res[cond]['z']['signed']
    out = []
    for i in (0, 1, 2, 4, 5, 6):
        core, ring = bd.lesion_masks(truth_c, i)
        out.append(bd.mixing_fit(S, truth['S_ideal'], core, ring))
    mix[cond] = out
    print(f"{label:38s} " + ' '.join(f"{o['a']:6.2f} / {o['rel_resid']:4.2f}" for o in out))"""))
C.append(code(r"""fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
for ax, j, name in zip(axes, (3, 4, 0, 1), ('L5 10 mm short', 'L6 6 mm short', 'L1 10 mm long', 'L2 6 mm long')):
    m0 = mix['noisy:E2:CAL:def'][j]
    ax.plot(1000 * TI, m0['ideal_core'], 'k-o', lw=2, label='ideal lesion')
    ax.plot(1000 * TI, m0['ideal_ring'], 'k--', lw=1, label='ideal WM ring')
    for _, c in LADDER:
        ax.plot(1000 * TI, mix[c][j]['rec_core'], '-o', ms=3, lw=1, label=res[c]['label'][:2].strip())
    ax.set_title(name, fontsize=9); ax.set_xlabel('TI (ms)'); ax.grid(alpha=0.3)
axes[0].set_ylabel('S(TI) / S(TI800), lesion mean'); axes[-1].legend(fontsize=7)
plt.tight_layout(); plt.show()"""))

C.append(md(r"""## 9. How much blur would explain it?

The ideal signal blurred with a Gaussian point-spread function (FWHM in voxels along read, pe1,
pe2) and fitted: the lesion bias that pure extra partial volume of a given width causes. WM is
unaffected by blur (its T1 varies smoothly)."""))
C.append(code(r"""fws = [(0, 0, 1.0), (0, 0, 1.5), (1, 1, 0), (2, 2, 0), (1, 1, 1.0), (1.5, 1.5, 1.0), (2, 2, 1.0), (3, 3, 1.0)]
print(f"{'blur FWHM (read, pe1, pe2) vox':32s} " + ' '.join(f'{"L"+str(i+1):>7s}' for i in (0, 1, 2, 4, 5, 6)) + '   (centred masks)')
for fw in fws:
    t1, _, _ = pu.fit_t1_grid(bd.blur_ideal(truth['S_ideal'], fw), TI, TR, MASK)
    rr = pu.evaluate_t1(t1, truth_c)
    print(f"{str(fw):32s} " + ' '.join(f"{rr[i]['bias_vs_ideal_pct']:+6.1f}%" for i in (3, 4, 5, 7, 8, 9)))"""))

C.append(md(r"""## 10. Regularisation dose-response and convergence

Bias vs λ for CALIPR on noise-free data with the true maps (λ = 0.005 default, 0.0005, 0.00005,
and λ → 0), and the λ → 0 solution after 100 vs 300 CG iterations (a change between them means
the unregularised problem is not converged and early stopping is still acting as a
regulariser)."""))
C.append(code(r"""dose = [('0.005', 'nf:true:CAL:def'), ('0.0005', 'nf:true:CAL:lam0.0005'), ('0.00005', 'nf:true:CAL:lam5e-05'),
        ('→0 (100 CG)', 'nf:true:CAL:zero'), ('→0 (300 CG)', 'nf:true:CAL:zero300')]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for i in (3, 4, 5, 7, 8, 9):
    ax[0].plot(range(len(dose)), [res[c]['rows_c'][i]['bias_vs_ideal_pct'] for _, c in dose], 'o-', label=ideal_rows[i]['region'])
ax[0].set_xticks(range(len(dose))); ax[0].set_xticklabels([d for d, _ in dose]); ax[0].set_xlabel('λ')
ax[0].set_ylabel('bias vs ideal (%, centred masks)'); ax[0].axhline(0, color='k', lw=0.8); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
ax[1].plot(range(len(dose)), [res[c]['rows'][0]['sd_ms'] for _, c in dose], 'o-', label='WM')
ax[1].plot(range(len(dose)), [res[c]['rows'][1]['sd_ms'] for _, c in dose], 'o-', label='GM')
ax[1].set_xticks(range(len(dose))); ax[1].set_xticklabels([d for d, _ in dose]); ax[1].set_ylabel('T1 SD within tissue (ms)')
ax[1].set_title('spatial T1 variation (true variation + error), noise-free'); ax[1].grid(alpha=0.3); ax[1].legend()
plt.tight_layout(); plt.show()
a, b = res['nf:true:CAL:zero']['z']['signed'], res['nf:true:CAL:zero300']['z']['signed']
br = truth['label'] >= 4
print(f'λ→0: relative change of the signed signal in brain, 100 → 300 CG iterations: '
      f'{np.linalg.norm(a[br] - b[br]) / np.linalg.norm(b[br]):.4f}')"""))

C.append(md(r"""## 11. Conclusions

_(written after the runs; see the next cell)_"""))

nb = nbf.v4.new_notebook(); nb['cells'] = C
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, sys.argv[1] if len(sys.argv) > 1 else 'CALIPR_LesionBias_V1.ipynb')
