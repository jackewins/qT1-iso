# Reconstruction comparison — brief for the method-development agents

## Goal

Quantitative T1 mapping at 64 mT (Hyperfine) from **undersampled 3D IR-FSE**: 4 inversion
times (TI = 30.75, 150, 400, 800 ms; TR = TI + 1.2 s) acquired in each of 3 orientations
(axial, coronal, sagittal) at 1.8 x 1.8 x 5 mm, R ≈ 6 (axial TI31 R ≈ 5). The per-plane
reconstructions are later combined into a 1.8 mm isotropic T1 map (Deoni et al., MRM
2022;88:1273 — `MRM-88-1273.pdf`).

The reconstruction being tuned now is **joint locally low rank (LLR)** across the 4 TIs
(`pics -R L`, see `synth/pipeline_utils.py: LLR_CMD`). To justify it we need two
alternatives built on identical inputs:

1. **CALIPR-style subspace reconstruction** — images constrained to a low-dimensional
   temporal subspace with spatial wavelet regularisation of the subspace coefficients
   (Dvorak et al., Sci Adv 2023, doi:10.1126/sciadv.adh9853). Reference implementation for
   multi-echo T2 (80 echoes): the `pics -B <basis> -R W:7:0:<lambda>` pattern.
2. **Model-based T1 reconstruction** — T1 (and M0) estimated directly from k-space with
   BART `moba` (Wang et al., MRM 2018;79:730, doi:10.1002/mrm.26726).

**Scope of this task: build a rough, working framework and run one end-to-end validation
on the synthetic phantom. Do NOT tune parameters or run parameter sweeps** — tuning is
done later together with the researcher.

## Data: synthetic phantom only

No real subject data are available to you, and none should be requested.
`synth/phantom.py` generates a 3-plane, 4-TI dataset with the real protocol's matrices,
voxel sizes, TIs/TRs and **real sampling coordinates** (`synth/acq_spec/`):

```
export BART_TOOLBOX_PATH=/path/to/bart
python synth/phantom.py /some/work/dir        # ~10-20 min, several GB RAM
```

* Anatomy: nested head ellipsoids (scalp fat, skull, CSF, cortical GM, WM), ventricles,
  deep GM, and 8 spherical WM lesions: diameters 2, 4, 6, 10 mm at T1 = 338 ms (+30 %)
  and 182 ms (-30 %) vs WM mid-range 260 ms. Lesion M0 = WM M0 (T1 contrast only).
* T1 varies smoothly in space within: WM 240-280, GM 240-380, CSF 3400-3700 ms
  (fat 180 ms and bone are placeholders).
* 8 synthetic coils, smooth background phase, a per-scan global phase offset
  (SD 5°, as measured between TI scans on real data), extra phase on echo group 1.
* Complex Gaussian noise, SD = 0.18 x RMS of that plane's TI800 k-space (measured on
  real data).
* Simulated on a finer grid (factor 2 x 2 x 4 in read, pe1, pe2) than the reconstruction
  grid to avoid the inverse crime.
* Ground truth: `truth_<PLANE>.npz` on each plane's reconstruction grid (label, T1_ms,
  M0, lesion_id, lesion_fraction = partial-volume fraction, true coil maps `sens`),
  `truth_iso.npz` on a 1.8 mm isotropic grid, `phantom_config.json`.
* `phantom.load_scans(out_dir)` returns `{plane: [ {name, stem, info}, ...by TI ]}`.
* **Motion (optional):** `python synth/phantom.py <dir> --motion` applies known rigid head
  motion per scan (`MOTION_PRESET` in `phantom.py`): drifts within each plane (<= 1 mm,
  <= 0.7 deg), a 2 mm step before axial TI800, and 1.5-3 mm / 1.5-2 deg offsets between planes.
  Coils stay fixed to the scanner (the head moves inside the coil). Poses are saved in
  `phantom_config.json` (object-to-scanner 4x4 per scan); each plane's truth is at its TI800
  pose (`truth_<PLANE>.npz['pose_ref']`), and `truth_iso.npz` is the unmoved head, so any
  registration can be scored against the known transforms. Without `--motion` the head is
  static (all numbers below are for the static phantom).

Note: every lesion is at or below the 5 mm slice thickness in at least one direction of
every plane, so partial volume is expected, especially for 2 and 4 mm.

### Evaluating against the truth

`pipeline_utils.evaluate_t1(T1_s, truth)` / `print_eval()` report, per tissue and per
lesion: the phantom T1 (`true`), the **resolution-limited reference** (`ideal`: T1 fitted
to the true signal with k-space truncated to the encoded extent, fully sampled and
noise-free, i.e. partial volume only) and the estimate. **Bias vs ideal is the
reconstruction's own error** and is the number to compare methods on. Check the phantom
with `python synth/check_phantom.py <dir>` (writes check_*.png).

### Reference result: current LLR method, coronal phantom plane

`python synth/run_llr_reference.py <dir> COR` (shared-TI self-calibrated maps, 2 map sets,
`pics -e -S -N -R L:7:7:0.006 -i 80 -b 4 -U`, both echo groups, PSIR, grid fit;
~16 min on a 10-core laptop):

| region | true | ideal | LLR | bias vs ideal |
|---|---|---|---|---|
| WM | 259.8 | 259.8 | 258.6 | -0.4 % |
| GM | 309.2 | 308.8 | 307.1 | -0.5 % |
| CSF | 3543 | 3863 | 190 | -95 % (PSIR polarity failure, see below) |
| lesion 10 mm, long T1 | 338 | 313.7 | 293.5 | -6.4 % |
| lesion 6 mm, long | 338 | 304.2 | 278.6 | -8.4 % |
| lesion 4 mm, long | 338 | 293.1 | 275.6 | -5.9 % |
| lesion 2 mm, long | 338 | 268.3 | 289.1 | +7.7 % (1 voxel, PV 0.19) |
| lesion 10 mm, short T1 | 182 | 203.9 | 230.2 | +12.9 % |
| lesion 6 mm, short | 182 | 211.5 | 231.3 | +9.4 % |
| lesion 4 mm, short | 182 | 225.1 | 259.8 | +15.4 % |
| lesion 2 mm, short | 182 | 238.9 | 243.6 | +2.0 % (1 voxel, PV 0.19) |

LLR pulls lesion T1 towards WM beyond partial volume. **CSF**: PSIR uses the TI800 image as
phase reference, and CSF (T1 ≈ 3.5 s) is still inverted at 800 ms, so its polarity is
flipped at every TI and the fit returns a short T1. This is a known limitation of the
PSIR step (not of the reconstruction); report CSF but do not try to fix it in this task.

## Shared conventions (use `synth/pipeline_utils.py`; do not re-implement)

* BART dims: k-space (1, read, pe, coil, 1, TI), trajectory (3, read, pe, 1, 1, TI).
  TIs in dim 5 (TE_DIM). `stack_tis()` stacks one plane/echo and returns a zero-weight
  pattern `P` when PE counts differ (axial).
* **Always pass trajectory/pattern as wrapper kwargs**: `bart(1, cmd, D, S, t=T, p=P)`.
  Never end a command string with a bare `-t`.
* Echo groups e = 0 and e = 1 are reconstructed separately (different phase), then the
  signed images of both are averaged before fitting.
* Coil maps: `shared_selfcal(scans_plane, e, nmaps=2)` — ENLIVE, coils shared across TI,
  two map sets (soft-SENSE). Maps are (X, Y, Z, coil, maps). Use these for all methods
  (plus, optionally, the true maps `truth_<PLANE>.npz['sens']` as an oracle check).
* Signed signal for PSIR: `combine_maps()` phase-references each map set to its own
  reference-TI (TI800) image and sums → take the real part.
* T1 fit: `fit_t1_grid()` (closed-form M0, dense T1 grid, V11) on the signed signal with
  S(TI) = M0 (1 - 2 e^(-TI/T1) + e^(-TR/T1)).

## Method notes

### CALIPR-style subspace
* 4 TIs only. SVD of simulated IR curves (these TIs/TRs, T1 100-4000 ms) needs K = 3 basis
  vectors to keep curves within ~3 % (K = 2: 15 % worst-case error). So the subspace
  itself only reduces 4 → 3 unknowns per voxel; the spatial regulariser does most of the
  work. Start with a dictionary basis, K = 3.
* The per-TI phase offsets and two map sets break a purely real basis. Decide and document
  how the first version handles this (e.g. per-TI global phase estimated from a low-res
  reconstruction and demodulated, or a complex/data-driven basis); list alternatives.
* Basis file layout for `pics -B`: TI along dim 5, coefficients along dim 6.

### Model-based (`moba`)
* With TR = TI + c, the IR equation is exactly the 3-parameter Look-Locker form that
  `moba -L` fits: S = M_ss - (M_ss + M0') e^(-TI R1*) with M_ss = M0,
  M0' = M0 (1 - e^(-c/T1)) and R1* = 1/T1 (no Look-Locker flip-angle correction applies).
  Verify this numerically first. Note c = 1.194 s for TI31 (TR 1.225 s) and 1.200 s for the
  others — quantify the model error this causes.
* `moba` assumes one complex image per voxel shared by all TIs: the per-TI phase offsets
  must be handled (document how). It returns parameter maps per plane (not TI images), so
  note what this means for the later 3-plane super-resolution step.
* Pass the shared self-calibrated coil maps if `moba` accepts them (last argument),
  otherwise document what `moba`'s internal coil estimation does.

## Deliverables (per method)

* Work on your own branch off `recon-comparison`: `recon-calipr` or `recon-moba`.
* One notebook in the repo root: `Recon_CALIPR_V1.ipynb` or `Recon_MOBA_V1.ipynb`,
  in the same style as `T1map_3plane_SR_V1.ipynb` (markdown explaining each step and
  why; configuration cell at the top; results cached outside the repo).
* End-to-end validation on the **coronal** phantom plane (both echo groups), then at least
  a run on all three planes if time permits:
  - images at all 4 TIs, T1 map, residual/R² map;
  - mean ± SD T1 in WM, GM, CSF (truth label masks, eroded) and **per lesion** vs truth;
  - the same numbers for the LLR reference (`llr_recon` with `LLR_CMD`) on the same
    phantom, for context only (no tuning of either);
  - runtime.
* Commit code and the executed notebook (phantom figures only), never phantom data
  (see `.gitignore`); push your branch.
* Finish with a short report (in the notebook's last markdown cell and as your final
  message): what was built, design decisions and open choices, the validation numbers,
  known limitations, and the parameters that will need tuning.

## Environment

* BART v1.0.00 or later (needs `ncalib`, `moba`, `pics -B`). If not installed:
  `git clone https://github.com/mrirecon/bart && cd bart && git checkout v1.0.00 && make -j`
  (needs gcc, make, libfftw3-dev, liblapacke-dev, libopenblas-dev, libpng-dev), then
  `export BART_TOOLBOX_PATH=$PWD`.
* Python: numpy, scipy, matplotlib, nbformat/nbconvert, jupyter.
