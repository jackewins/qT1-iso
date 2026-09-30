"""Executed cells 0-23 (partial run) + remaining unexecuted cells + final report cell."""
import sys
import nbformat as nbf

REPORT = r"""
## 17. Report (V1, status at hand-over; work stopped early at the coordinator's request)

**State of this notebook.** Sections 1-10 (tables) were executed on the coronal plane, both
echo groups; their outputs are shown above. **Section 10 figures and sections 11-15 were
written but not executed** (no outputs): bias bar chart, T1-vs-truth figure, model-generated
TI images and k-space residual, M0'/M_ss consistency, coil-map QC, LLR side-by-side,
all-plane run. They have not been debugged; expect small fixes on first run. The LLR
reference was **not** re-run here. Everything ran on the researcher's laptop next to other
heavy BART jobs, so runtimes are inflated.

### What was built
* `moba -L` per plane and echo group from the stacked 4-TI k-space (`pu.stack_tis`, TIs in
  dim 5, trajectory as wrapper kwarg `t=T`), per-TI global phase demodulation, T1 = 1/R1*,
  echo-group combination, evaluation with `pu.evaluate_t1`.
* Reconstruction is **slice-wise along the fully sampled Cartesian readout** (readout iFFT,
  then 112 independent 2D `moba` problems for coronal, 6 in parallel). Cache: `PHANTOM_DIR/moba/`.

### Model-mapping verification (section 3)
* With TR = TI + c, S = M0(1 - 2e^(-TI/T1) + e^(-TR/T1)) is **exactly** moba's
  S = M_ss - (M_ss + M0')e^(-TI R1*), with M_ss = M0, M0' = M0(1 - e^(-c/T1)), R1* = 1/T1.
  The max difference is 4e-16 M0 over T1 0.1-4 s. There is no Look-Locker correction, so
  do not use `looklocker`.
* c = 1.194 s at TI31 vs 1.200 s gives a T1 error (noise-free, 3 free parameters) of
  **< 0.07 % for 180-340 ms** (all brain tissue and lesions), 1.2 % at 1 s and **+11 % for
  CSF (3.5 s)**. With TIs <= 0.8 s the 3-parameter fit is ill-conditioned for long T1.
* TI file: complex, shape (1, 1, 1, 1, 1, 4), in **seconds**, so R1* is in 1/s.

### Design decisions and open choices
* **Axial padding / `-p`:** `moba -p` is a PSF for pre-gridded Cartesian data, not a sample
  weight, so it is not passed. It is not needed either: with `-t`, moba weights only
  non-zero samples (`estimate_pattern`). Section 4 confirms that the padded samples are
  exactly the all-coil-zero samples.
* **Coil maps:** moba does **not** accept external maps. The last positional argument is an
  *output*. `--other ksp-sens` is only an initial value in moba's Sobolev k-space
  representation and is still updated, and `-L` ignores `no-sens-deriv`. moba estimates
  the coils jointly with the maps: shared across TIs, Sobolev-smooth, **one map set only**.
  So the 2-map `shared_selfcal` maps cannot be used. Open option: pass
  `shared_selfcal(nmaps=1)` as the initial value via `ksp-sens`.
* **Per-TI phase:** doubled-angle estimate from per-coil tapered gridding images, then
  demodulation in k-space. Against the true offsets (up to 11°) the error is <= 0.3° at
  TI31/TI400 and 0.8-1.2° (systematic) at TI150. Alternatives are in section 5.
* **Echo groups:** reconstructed separately, then T1 = 1/mean(R1*) with equal weights.
* **3D vs slice-wise:** one 3D call with the same options failed (coronal echo 0, 35 min,
  ~7 GB). The residual stayed at 51 % of the data norm (noise floor about 24 %) and WM T1
  came out at 90 ms. The fixed-norm scaling does not adapt to volume size.

### Options (all untuned)
`moba -L -l1 -i 8 -C 100 -j 0.01 --normalize_scaling --scale_data 5000 --scale_psf 1000`,
default init (M_ss = M0' = 1, R1* = 1/s), `--img_dims 1:ny:nz` per readout slice.

### Validation: coronal plane, echo groups combined

| region | true | ideal | moba mean | bias vs ideal |
|---|---|---|---|---|
| WM | 259.8 | 259.8 | 607.9 (median 250.2) | +134 % (median -3.6 %) |
| GM | 309.3 | 308.8 | 654.6 (median 289.7) | +112 % (median -5.9 %) |
| CSF | 3544 | 3819 | outliers (median 313) | fails |
| lesion 1, 10 mm, long T1 | 338 | 313.7 | 327.5 | +4.4 % |
| lesion 2, 6 mm, long | 338 | 304.2 | 258.2 | -15.1 % |
| lesion 3, 4 mm, long | 338 | 293.1 | 463.8 | +58.2 % |
| lesion 4, 2 mm, long | 338 | 268.3 | 343.5 | +28.0 % (1 voxel) |
| lesion 5, 10 mm, short T1 | 182 | 203.9 | 201.5 | -1.2 % |
| lesion 6, 6 mm, short | 182 | 211.5 | 191.3 | -9.5 % |
| lesion 7, 4 mm, short | 182 | 225.1 | 215.1 | -4.5 % |
| lesion 8, 2 mm, short | 182 | 238.9 | 671.7 | +181 % (1 voxel) |

Medians over the eroded masks: WM 250 ms (IQR 199-325), GM 290 ms (IQR 194-474).
**V1 is not usable yet.** The means are dominated by voxels where R1* goes to 0 (T1 > 1 s in
3.8 % of WM and 14 % of GM), and voxel-wise noise is very large (echo 1 vs echo 0 T1 IQR
±38 %). The solver shows why: in many readout slices the IRGNM **diverges**. The residual
entering the last Newton step reaches up to 22x (echo 0) and 46x (echo 1) the data norm
(median 0.58 / 0.64; noise floor about 0.24), mostly in slices towards the head periphery
(x about 24-32 and 88-95). Central slices converge: slice 76 has residual 0.34 and WM/GM
257 / 312 ms vs ideal 261 / 312. The lesion rows mostly reflect noise, not regularisation
bias. A comparison with LLR is not meaningful yet (LLR in the brief: WM -0.4 %, GM -0.5 %).

**Runtime:** 6.6 min (echo 0) and 8.5 min (echo 1) wall time with 6 parallel single-thread
jobs on a loaded 10-core laptop. That is about 15 s per 2D slice single-threaded. The
phase estimate takes about 5 s per echo.

### Known limitations / not done
* The solver is unstable in peripheral slices; this is the first thing to fix. A diagnostic
  run on a diverging slice (x = 26, `-d 5`) was started but stopped before it produced output.
* Section 10 figures and sections 11-15 are not executed. AX and SAG are not reconstructed,
  and the LLR reference was not re-run.
* CSF: the model is ill-conditioned at these TIs (+11 % from the c mismatch alone), on top
  of the instability.
* Slice-wise 2D: regularisation and coil smoothness act only within each (pe1, pe2) plane,
  and each slice has its own normalisation (handled by `rel_scale`).
* M0' is free (not tied to c and T1), so moba fits 3 parameters where the PSIR fit uses 2.

### Parameters to tune (with the researcher)
1. Data/PSF scaling (`--scale_data`, `--scale_psf`, `--normalize_scaling`): sets the effective
   regularisation. It must adapt to volume/slice size (this would also allow 3D).
2. `-j` (alpha_min) and `-R` (regularisation strength and schedule).
3. `-i` Newton steps and `-C` inner iterations (the divergence happens at late, weakly
   regularised steps).
4. `-s` FISTA step, `-B` lower bound on R1*, `--other pinit` (initial values), and
   `--other pscale` (balancing R1* against M_ss/M0', as in Wang 2018).
5. Regulariser type (`-l1`, `-l2`, or `-r` terms per map).
6. Coil initialisation (`ksp-sens` from `shared_selfcal(nmaps=1)`) and Sobolev coil parameters.
7. `PHASE_TAPER` (minor) and echo-group weighting.
"""

done = nbf.read(sys.argv[1], 4)          # partially executed copy (run_partial.py output)
full = nbf.read('Recon_MOBA_V1.ipynb', 4)
assert len(done.cells) == 24
cells = done.cells + full.cells[24:-1] + [nbf.v4.new_markdown_cell(REPORT.strip())]
cells[0].source += ('\n\n> **Status (V1, hand-over):** partially executed. Sections 1-10 have outputs '
                    '(coronal); the rest is written but not run. See section 17.')
full.cells = cells
nbf.write(full, 'Recon_MOBA_V1.ipynb')
print(len(cells), 'cells')
