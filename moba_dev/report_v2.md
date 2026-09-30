## 18. Report (V2: first stable version)

**State.** Every section of this notebook was executed on the static phantom, coronal plane,
both echo groups (sections 1-15; section 16, axial and sagittal, is switched off and was not
run). All numbers below are from the outputs above. Phantom only; no subject data.

### What was wrong in V1, and what V2 changes (section 7)

Measured with the raw-data residual after the last Newton step (moba's own log shows the
residual *entering* each step, which hid it), V1 was broken in 10 of 13 screened readout slices,
central ones included (residual 3-90× the noise floor, 4-19 % of brain voxels with R1* at 0).
Three causes, each shown on single readout slices:

1. **Null space at the R1* bound + no step control.** At R1* = 0, ∂S/∂M_ss = 0, so M_ss is held
   only by the weak late-step wavelet penalty; IRGNM-FISTA has no proximal term on the maps, so
   late steps (α → 0.01, 100 FISTA iterations) can jump. → needs enough α_min.
2. **Per-slice data normalisation** (`--normalize_scaling`) set every slice's data norm to 5000,
   so noisy peripheral slices got ~2.5× less regularisation relative to their noise. → V2 uses a
   **fixed, noise-referenced scale per plane**: `--scale_data 4.545/σ` (σ = k-space noise SD;
   45.0 in the coronal plane) and `--scale_psf` = moba's own PSF normalisation (1000/‖PSF‖).
3. **The default coil model is too smooth.** With `sobolev_a = 880` moba cannot represent the
   coils (10-13 % error for the true coils; the rest goes into the maps: WM at the top of the head
   194 vs 262 ms, image intensity halved there). Per-TI SENSE with the true coils gets those slices
   right. → **`--sobolev_a 220`** (4-6 % representation error).

V2: `moba -L -l1 -i 10 -C 100 -j 0.3 --sobolev_a 220 --scale_data 4.545/σ --scale_psf 1000/‖PSF‖`,
slice-wise along the readout (as V1), default initialisation, echo groups combined as
T1 = 1/mean(R1*). α_min = 0.3 is conservative: 0.1 diverges in 6 of 26 screened slice runs.
Levers that did **not** help: lower bound `-B`, l2 on M_ss, `-N`, initial R1* = 3 s⁻¹ (helps the
top of the head but biases the long-T1 lesion and does not survive more iterations),
`--pusteps/--ratio`, `-C 200` (destabilises a borderline slice). `--other pscale` is not a valid
lever for `-L` in BART v1.0.00 (the model ignores it; only the output is rescaled). Oracle coil
initialisation via `ksp-sens` (conversion implemented and verified) helped much less than the
coil-model change.

### Validation: coronal plane, echo groups combined (section 11)

**Convergence:** 0 of 224 slice runs flagged (raw residual > 1.5× noise floor); median 0.88×
noise floor. Only the scalp-level slices x = 7-15 (and x = 106, echo 1) sit at 1.0-1.27×: part of
the scalp/fat signal at the bottom of the head is not fitted (visible as the band in the residual
image, section 12); no brain in those slices. Global scale check a = 1.0075 (should be 1):
the scaling bookkeeping is right. Echo 1 vs echo 0 T1: median +0.2 %, IQR ±15 % (V1: ±38 %).

Evaluation (`pu.evaluate_t1`; means over the eroded tissue masks and the lesion masks). **LLR**
is the reference method re-run here on the same phantom (`synth/run_llr_reference.py`, unchanged
`LLR_CMD`); it reproduces the table in `RECON_COMPARISON.md` to 0.1 ms, i.e. the regenerated
phantom is identical.

| region | n | ideal (ms) | LLR (ms) | moba V2 (ms) | LLR vs ideal | **moba V2 vs ideal** | ± SE of moba mean* |
|---|---|---|---|---|---|---|---|
| WM | 70965 | 259.8 | 258.6 (SD 20) | 259.2 (SD 31) | −0.4 % | **−0.2 %** | < 0.1 % |
| GM | 3750 | 308.8 | 307.1 (SD 31) | 301.5 (SD 38) | −0.5 % | **−2.3 %** | 0.2 % |
| CSF | 387 | 3867 | 190 | fails (median 1.8 s) | −95 % (PSIR polarity) | fails (ill-conditioned) | – |
| lesion 1, 10 mm, long T1 | 34 | 313.7 | 293.4 (SD 31) | 306.4 (SD 42) | −6.5 % | **−2.3 %** | 2.3 % |
| lesion 2, 6 mm, long | 10 | 304.2 | 278.5 (SD 5) | 291.2 (SD 35) | −8.5 % | **−4.3 %** | 3.6 % |
| lesion 3, 4 mm, long | 4 | 293.1 | 275.6 (SD 3) | 301.2 (SD 17) | −5.9 % | **+2.8 %** | 2.8 % |
| lesion 4, 2 mm, long | 1 | 268.3 | 289.1 | 281.4 | +7.7 % | **+4.9 %** | 1 voxel |
| lesion 5, 10 mm, short T1 | 34 | 203.9 | 230.2 (SD 15) | 209.9 (SD 28) | +12.9 % | **+2.9 %** | 2.4 % |
| lesion 6, 6 mm, short | 7 | 211.5 | 231.3 (SD 8) | 224.0 (SD 20) | +9.4 % | **+5.9 %** | 3.6 % |
| lesion 7, 4 mm, short | 4 | 225.1 | 259.8 (SD 5) | 236.2 (SD 26) | +15.4 % | **+4.9 %** | 5.8 % |
| lesion 8, 2 mm, short | 1 | 238.9 | 243.6 | 265.5 | +2.0 % | **+11.1 %** | 1 voxel |

\* SD/√n relative to ideal, assuming independent voxels — a lower bound, since the regulariser
correlates neighbouring voxels. 1-voxel lesions are single noisy samples for both methods.

**Reading.** In WM, V2 matches the ideal as well as LLR; GM is 2.3 % low (LLR −0.5 %). For the
lesions, V2's bias vs ideal is smaller than LLR's for the 10, 6 and 4 mm lesions of both
polarities (e.g. 10 mm: −2.3 vs −6.5 % and +2.9 vs +12.9 %): LLR pulls lesions towards WM,
V2 much less. But V2 is **noisier** (voxel SD ~1.5× LLR's in WM, lesion SDs 1.4-7× LLR's), so for
the 6 and 4 mm lesions the differences amount to only 1-3 standard errors (a lower bound, see *),
and the 2 mm lesions are single noisy voxels. The fair comparison is at matched noise (or matched
resolution) — part of the tuning (stages 2 and 5). Qualitatively (section 15 figure): LLR is
smoother but its errors are *structured* (a low-T1 streak region around the left deep GM;
ventricles inverted by the PSIR polarity), whereas V2's error is unstructured, voxel-scale
noise, with the ventricles correctly long-T1 and the deep-GM nuclei faintly visible.

WM and GM **medians** (eroded masks): WM 256.1 ms vs ideal 259.6 (−1.3 %), IQR 237-277,
robust SD 29 ms; GM 298.9 vs 308.0 (−3.0 %), robust SD 37 ms. No voxel of the eroded WM and GM
masks has T1 > 1 s (V1: 3.8 % of WM, 14 % of GM).

### Other checks

* **Model consistency (section 13):** M0'/M_ss − (1 − e^(−1.2 s·R1*)) has median −0.01 (WM) and
  +0.03 (GM), IQR 0.15: the free M0' agrees with the known recovery time in tissue; not in CSF
  (+0.32), see limitations.
* **Coils (section 14):** moba's coils correlate 0.96-0.99 with the true maps per coil (|map|
  inside the head), better than `shared_selfcal(nmaps=1)` (0.93-0.97).
* **Data consistency per TI (section 12):** residual at or below the noise floor at every TI.

### Known limitations

* **CSF fails.** Median T1 1.8 s vs ideal 5.0 s (eroded CSF mask; the mean is meaningless, a few
  voxels have R1* ≈ 0). With TIs ≤ 0.8 s a CSF curve is still almost linear, the 3-parameter model
  is ill-conditioned there (section 3: +11 % from the TI31 recovery-time mismatch alone), and the
  joint wavelet likely pulls the thin CSF layers towards the surrounding tissue (not tested). (LLR's CSF failure has a different
  cause, the PSIR polarity.) CSF is not a target of this protocol; flagged, not fixed.
* **Noise.** WM robust SD 29 ms (11 %) with α_min = 0.3 — the price of stability and of fitting
  3 parameters where the PSIR fit uses 2. Regularisation strength is tuning stage 2.
* **GM −3 %** and a small WM deficit in the central pe2 slices (median −7 to −11 ms at z = 21-23):
  most likely regularisation towards the surroundings (not isolated yet); revisit with the regulariser.
* **Slice-wise 2D:** no regularisation along the readout; each readout slice is its own problem
  (visible as horizontal streaks in the background, not in tissue). 3D is not retried yet.
* **One noise realisation, static phantom only**; the stability margin (α_min 0.1 → 0.3) is from
  13 screened slices and the full plane of this realisation.
* The per-TI phase estimate carries a systematic ~1° error at TI150 (section 5); not revisited.

### Runtime (4-core container, 4 slices in parallel, 1 thread each)

moba V2: 18.6 min (echo 0) + 16.2 min (echo 1) for the coronal plane (~0.6 core-min per slice);
the whole notebook incl. the cached section-7 diagnosis ~40 min. LLR reference: 35 min (both echo groups, incl. two `ncalib` runs; `pics` 14.5 min per echo group).
Section 7 from scratch: ~25 min (all runs are cached in `PROC_DIR/diag/cache`).

TUNING_PLAN
