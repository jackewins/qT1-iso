## 10. Result and decision (stage 1)

**Phantom v2, coronal plane, all other options as V2.** Every number below is from the outputs
above.

### Stability boundary

| α_min | screen: flagged slice runs (of 26) | full plane: flagged slice runs (of 224) | worst residual / noise floor |
|---|---|---|---|
| 0.10 | 8 (x = 48, 56, 64, 96 in **both** echo groups) | – | 868 (screen) |
| 0.15 | 1 (x = 56, echo 1) | **15** (10 distinct slices) | 5830 |
| 0.20 | 0 | **0** | 1.10 |
| 0.25 | 0 | – (screen only) | 1.08 (screen) |
| 0.30 | 0 | **0** | 1.27 |

* The boundary lies **between 0.15 and 0.2** on phantom v2 (on v1 it was between 0.1 and 0.3,
  not resolved further). At 0.2 every one of the 224 slice runs converges.
* **The screen under-calls instability**: at 0.15 it flagged 1 of 26 slice runs, the full plane
  15 of 224 (6 in echo 0, 9 in echo 1; 10 distinct slices: x = 53-57, 59, 62, 94, 102, 103).
  A 13-slice screen is a cheap first filter, not a stability proof; the full-plane residual QC is
  the real test.
* Divergence is **slice-specific, not random**: at 0.1 the same four slices fail in both echo
  groups (independent noise), and the 0.15 failures cluster in the same region
  (x = 54-64, 94, and two scalp-level slices). Anatomy/coil geometry decide where the
  linearised steps go wrong.

### What lower α_min buys (0.2 vs 0.3)

| region | ideal (ms) | α_min 0.3 | α_min 0.2 |
|---|---|---|---|
| WM | 274.7 | 274.8 (+0.0 %) | 276.9 (+0.8 %) |
| GM | 339.3 | 329.4 (**−2.9 %**) | 335.9 (**−1.0 %**) |
| lesion 10 mm, long T1 | 340.7 | −1.6 % | +0.7 % |
| lesion 6 mm, long | 336.2 | −4.5 % | −5.3 % |
| lesion 4 mm, long (2 voxels) | 331.9 | +0.5 % | +5.2 % |
| lesion 2 mm, long (2 voxels) | 283.5 | +7.7 % | +9.3 % |
| lesion 10 mm, short T1 | 207.0 | +3.4 % | +3.4 % |
| lesion 6 mm, short | 206.9 | +9.8 % | +8.9 % |
| lesion 4 mm, short (2 voxels) | 214.8 | +7.7 % | +6.4 % |
| lesion 2 mm, short (2 voxels) | 255.7 | +4.8 % | +7.9 % |
| WM robust SD (ms) | – | 32.4 | 40.9 (+26 %) |
| GM robust SD (ms) | – | 34.6 | 42.9 |

**0.15** converges in most slices but diverges in 10 (above). Without a fallback those slices ruin
the means (WM +20.7 %, GM +550 % vs ideal) although the medians stay sensible (WM 272.1, GM 334.9 ms;
robust SD 50.5 / 55.3 ms). With the fallback (next paragraph) its numbers are WM +1.3 %, GM −0.1 %,
lesions +2.2 / −6.0 / +8.3 / +11.2 % (long T1, 10/6/4/2 mm) and +3.7 / +8.1 / +5.3 / +10.6 % (short):
GM improves further, the lesions do not, and the noise rises again (WM robust SD 44.7 ms with the
fallback, 50.5 ms as reconstructed, vs 40.9 at 0.2).

Going from 0.3 to 0.2 removes most of the GM bias (−2.9 → −1.0 %) at ~26 % more noise; the
lesion means hardly move (the 10 mm lesions are within 2 % of ideal at both; the 4 and 2 mm
lesions now hold only 2 voxels each in v2's centred masks, so their ± 5 % differences are noise).
Visually (section 7) the maps at 0.2 are slightly noisier, not sharper: α_min is a weak knob for
image quality, because it is bounded below by stability. **Noise, not regularisation bias, is
what limits these maps now** (WM robust voxel SD 12-15 % of T1; the 4 and 2 mm lesions are not visible).

**Adaptive fallback** (re-do flagged slices at 0.3, section 8): at 0.2 it never triggers (0 slices);
at 0.15 it replaces 10 of 112 readout slices and turns a broken map into a usable one. It works as
a safety net, but lowering α_min below 0.2 buys only GM accuracy (−1.0 → −0.1 %) at 10-25 % more
noise and no lesion gain, so the agreed exception ("a lower α_min clearly reduces lesion bias")
does not apply.

**`-R 3`** (α decays 3× per Newton step instead of 2×, so more steps run at α_min) at 0.2:
**2 of 26** screened slice runs diverge (0 with the default `-R 2`). Faster decay destabilises; keep
`-R 2`.

### Decision (for the researcher)

By the agreed rule — smallest α_min with no flagged slice on the full plane (0.2), plus one grid
step of margin — the choice is **α_min = 0.25**. The alternative the rule allows is **0.2 with
the residual-QC fallback** (any slice above 1.5 × its noise floor is re-done at 0.3): it keeps
0.2's lower GM bias and gets its margin per slice from the QC instead of globally; on this phantom
the fallback would never trigger at 0.2. My recommendation is **0.2 + QC fallback**, because the
QC is also the safeguard needed on real data, where the boundary will move. Either way, stage 2
should decouple the regularisation *strength* from stability: `--l1val` scales only the
wavelet threshold, so it can trade noise against bias at a fixed, stable α_min.

### Next: stage 2 (proposal, awaits approval)

* **2a, wavelet strength at α_min 0.2 (+ QC fallback):** `--l1val` ∈ {0.5, 2, 4} (1 = current).
  Screen (≈ 5 min each, stability) then full planes for the values that pass (≈ 35-45 min
  each): numbers, maps, histograms as here. Expected: higher `l1val` lowers the noise and
  raises partial-volume-like lesion/GM bias; the researcher picks the trade-off by eye.
* **2b, regulariser type:** total variation on the maps (`-r T:6:0:λ`, ADMM path, 2-3 λ) vs
  the joint wavelet; screen first.
* LLR on phantom v2 is not re-run here (the brief's table is v1); for a side-by-side in stage 2
  it would need one run (≈ 35 min).
