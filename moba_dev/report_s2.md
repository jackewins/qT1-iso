## 8. Result and decision (stage 2a)

**Phantom v2, coronal plane, α_min 0.2, joint l1-wavelet; only `--l1val` varies.** Every number
below is from the outputs above (the fallback replaces flagged slices as described in section 3).

### Stability: two failure modes

| l1val | screen: flagged (of 26) | full plane: flagged slices (echo 0 / 1, of 112) | median residual / noise floor | failure mode |
|---|---|---|---|---|
| 0.5 | 8 | **37 / 36** (x = 34-69, 74-79, 91-103) | 0.92 | divergence (up to 47 000 × the noise floor) |
| 1 | 0 | 0 / 0 | 0.86 | – |
| 2 | 2 | 3 / 4 (x = 13-16) | 0.89 | under-fit, top of the head |
| 4 | 4 | 13 / 12 (x = 13, 15-26) | 0.94 | under-fit, top of the head |
| 8 | 0 | – (screen only) | 1.29 (screen) | under-fit in every slice |

* **Weaker than 1 is not usable at α_min 0.2.** l1val and α_min both set how strongly the maps are
  regularised; stage 1 put α_min at the stability boundary for l1val 1, so l1val 0.5 crosses it:
  a third of the plane diverges, more than the screen suggested (8 of 26). Even after the
  replacement, 12 GM/WM voxels in converged slices have R1* ≤ 0 or T1 > 1000 ms (0 at l1val ≥ 1).
* **Stronger than 1 fails differently:** in the low-signal slices at the top of the head the
  wavelet removes signal along with the noise, the fit stays 1.5-2 × above the noise floor and
  T1 there is wrong (a smooth ramp unrelated to the anatomy, e.g. 170 → 2800 ms over 12 voxels
  of slice x = 15 at l1val 4). No collapse,
  no divergence, but the QC catches it.
* **The median residual is the global ceiling** (discrepancy principle: a fit should explain the data
  to the noise level, ≈ 1 × the noise floor): 0.86 → 0.89 → 0.94 → 1.29 for l1val 1 → 2 → 4 → 8.
  l1val 4 is the strongest setting that still fits the bulk of the plane; 8 under-fits everywhere.
* **The fallback works for both modes:** each flagged slice is taken from the nearest setting towards
  l1val 1 that passes there. At l1val 4: x = 13, 17-26 from l1val 2 and x = 15-16 from l1val 1. The
  α_min 0.3 re-runs the runner also made would have failed: they stay diverged at l1val 0.5 (up to
  379 × the floor) and make the under-fit worse at l1val 2 (up to 3.6 ×).

### What the wavelet strength buys (after the fallback)

| | ideal | l1val 0.5 | l1val 1 (stage 1) | l1val 2 | l1val 4 |
|---|---|---|---|---|---|
| WM mean T1 bias | 274.7 ms | +2.4 %¹ | +0.8 % | +0.6 % | +0.8 % |
| GM mean T1 bias | 339.3 ms | +1.1 % | −1.0 % | −3.2 % | **−4.7 %** |
| WM robust SD (noise part) | – | 48.4 (45.0) ms | 40.9 (38.2) | 26.9 (24.5) | **17.0 (14.1)** |
| GM robust SD (noise part) | – | 51.1 (46.6) ms | 42.9 (38.3) | 32.0 (27.1) | 25.9 (20.2) |
| lesion 10 mm long T1 | 340.7 ms | +4.5 %² | +0.7 % | −4.0 % | −6.9 % |
| lesion 6 mm long | 336.2 ms | −7.6 %² | −5.3 % | −4.1 % | −4.4 % |
| lesion 10 mm short T1 | 207.0 ms | +1.9 %² | +3.4 % | +4.6 % | +8.5 % |
| lesion 6 mm short | 206.9 ms | +5.9 %² | +8.9 % | +11.6 % | +15.7 % |
| lesion-WM contrast kept, 10 mm long / short | 100 % | 114 / 104 %² | 100 / 93 % | 77 / 89 % | 61 / 77 % |
| voxel CNR, 10 mm long / short | – | 1.55 / 1.45 | 1.62 / 1.53 | 1.88 / 2.23 | **2.37 / 3.07** |
| voxel CNR, 6 mm long / short | – | 0.61 / 1.28 | 1.01 / 1.26 | 1.71 / 1.69 | **2.62 / 2.20** |
| WM dip at the central slices z = 21-23 | – | −9.8 ms | −8.8 ms | −1.9 ms | +0.4 ms |

¹ inflated by the 12 failed voxels (the WM median is 272.9 ms). ² most of the lesion band
(x = 62-79) diverged at 0.5 and was replaced by l1val 1 slices, so these are partly l1val 1 values.
The 4 and 2 mm lesions hold 2 voxels each (±5-10 % is noise); they are in the table above but not
used here.

* **Noise falls steeply, WM accuracy does not change:** WM robust SD 40.9 → 26.9 → 17.0 ms
  (−34 %, −58 %), almost all of it noise; WM bias stays at +0.6-0.8 %. The speckle (single voxels
  > 100 ms off: 8.7 % of brain voxels at l1val 1, 0.4 % at l1val 4) is what made the stage-1 maps
  look "very noisy"; it is noise, not an artefact (section 7, point 1).
* **The cost is contrast in small and thin structures:** GM (a thin ribbon after erosion, mixed with
  WM by the smoothing) −1.0 → −3.2 → −4.7 %; lesions are pulled towards WM (10 mm lesions keep
  77-89 % of their contrast at l1val 2, 61-77 % at 4; short-T1 lesions are hit harder: +8.5 % and
  +15.7 % at l1val 4 for 10 and 6 mm).
* **But noise falls faster than contrast:** the voxel CNR of every ≥ 6 mm lesion rises
  monotonically with l1val (1.0-1.6 at l1val 1, 1.7-2.2 at 2, 2.2-3.1 at 4). For *seeing* lesions
  stronger is better up to 4; for *measuring* lesion T1, weaker is better. The pre-agreed stage-2
  rule (lowest lesion bias with WM/GM within ± 2 %) picks l1val 1; it was written before noise was
  identified as the main problem.
* **l1val 0.5 buys nothing:** noisier, a third of the plane diverged, failed voxels.

### Artefacts (section 7)

1. **Speckle = noise** (see above): echo-group voxel errors correlate at 0.07 (l1val 1); the far-off
   voxels coincide between the echo groups at close to chance level (11 vs 9 %); 85 % are too long (T1 = 1/R1*
   turns symmetric R1* noise into a long tail). Stronger wavelet removes it.
2. **Central-slice dip (z = 21-23):** −9 to −10 ms at l1val ≤ 1, −1.9 ms at 2, gone at 4. The noise
   is *not* higher in those slices (37.1 vs 38.5 ms at l1val 1), so low SNR is ruled out; it is a
   systematic R1* offset (+0.2 s⁻¹ median at z = 22) that appears only with weak regularisation.
   My earlier hypothesis (wavelet grid locked to the centre + the fixed shift sequence) predicts a
   larger error with *stronger* thresholding, the opposite of what is seen, so it is now less
   likely. It is not excluded and the mechanism is open. At l1val ≥ 2 it is ≤ 2 ms.
3. **Lines along the readout** (vertical in these coronal maps): errors that repeat at the same
   (y, z) position in consecutive readout slices, e.g. z = 14, y = 38: +34 ms median over 40
   readout slices at l1val 1, +12 ms at l1val 4. Each readout slice is an independent 2D problem
   with the same geometry, the same sampling and the same wavelet shift sequence (moba resets the
   random-shift state to 1 at every Newton step, `src/moba/iter_l1.c`), so a position-locked error
   repeats along the readout. They shrink with l1val, like the dip. Not analysed further.
4. **Edges:** CSF (ventricles, the CSF rim, the top-of-head CSF at x = 13-14) is far too short in
   every setting (true 5000 ms is beyond the measured TI range); this is outside the brain masks.
   At l1val 4 deep GM next to the ventricles is pulled down (blue halos in the difference maps:
   smoothing across the ventricle edge).

### Decision (for the researcher)

**My recommendation is l1val 4 with the per-slice fallback (4 → 2 → 1)** as the working point
for the next stages, because it addresses the problem raised (noise and artefacting):
- the noise falls 2.7×;
- the speckle (8.7 → 0.4 %) and the central-slice dip are gone;
- lesion CNR is the highest in the grid;
- WM is unbiased;
- the plane still fits to the noise level (median 0.94).

It **violates the pre-agreed ±2 % GM criterion** (−4.7 %), and lesion T1 values are pulled towards
WM by 4-16 %. If lesion and GM T1 *values* are the endpoint, **l1val 2** is the compromise:
- noise falls by a third;
- GM bias is −3.2 %;
- lesions are within −4 / +12 %;
- only 3-4 slices need the fallback.

The final choice is yours from the maps and histograms (section 6).

Caveats:
- One noise realisation, one plane, a phantom whose SNR may differ from the scanner's.
- On real data the per-slice QC decides where l1val 4 under-fits; that safeguard comes with the
  fallback.

### Next (proposals, await approval)

* **Contrast recovery at the chosen l1val:** moba's wavelet is a *joint* group threshold over
  (M_ss, M0', R1*) (`COEFF_FLAG`, `src/moba/iter_l1.c`), so the numerical scale of R1* sets how much
  its own edges count. Scaling the TI axis by k (R1* → R1*/k; `diag_slice.run(ti_scale=k)`, the
  -L model ignores `--other pscale`) shifts that weight. Screen k ∈ {0.5, 2} at l1val 4 plus the
  lesion slab (≈ 2 × 13 min ≈ 25 min).
  - Tests whether R1* edges (lesions, GM) can be kept while M_ss/M0' stay smooth.
  - Decision: adopt k if lesion contrast rises with ≤ 10 % more WM noise and no new flagged slices.
  - Uncertain: k also changes the conditioning of the Gauss-Newton steps, so stability must be
    re-checked.
* **Grid- vs anatomy-locked errors (dip, lines):** shift the object by 2 pe2 and 2 pe1 voxels before
  reconstruction (k-space phase ramp), reconstruct the 13 screen slices at l1val 1 (≈ 5 min) and see
  whether the dip and the lines move with the grid or with the anatomy. Optionally, a BART build
  with a per-slice random-shift seed. Low priority if l1val ≥ 2 is chosen.
* **Stage 3 (iteration counts)** as planned, at the chosen l1val: `-i` ∈ {8, 10, 12, 14} ×
  `-C` ∈ {50, 100} on the lesion slab (≈ 70 min).
* LLR on phantom v2 for a side-by-side (≈ 35 min) is still open.
