### Tuning plan (proposal — each stage runs only after the researcher's go-ahead)

Order as suggested: stability → regularisation strength and type → iteration counts → coil
initialisation → image-quality sweeps. Costs on this 4-core container (one readout slice with
`-i 10 -C 100` ≈ 48 s single-threaded, 4 in parallel):

| unit | what | per configuration |
|---|---|---|
| **screen** | 13 readout slices (x = 8…104) × 2 echo groups: stability over the whole head | ≈ 5.5 min |
| **lesion slab** | the 18 readout slices through all 8 lesions (x = 62-79) × 2 echo groups: per-lesion bias vs ideal, WM/GM in the slab, figures | ≈ 7.5 min |
| **full plane** | 112 slices × 2 echo groups (section 8) | ≈ 46 min |

Every stage reports bias vs ideal (the key number) per tissue and per lesion, SD, the
per-slice residual/noise-floor QC, and runtime; grid figures as in
`T1map_3plane_SR_V1.ipynb` §13 (rows = the swept strength, columns = the second parameter):
T1 map through the lesion plane, a lesion zoom, the model-generated signed TI800 and TI150
images, and the error vs ideal, all with shared windows.

**Stage 1 — stability boundary (α_min).** *Tests:* the smallest α_min at which every slice
converges, i.e. how much regularisation stability forces on us (V2's 0.3 is conservative:
0.1 diverges in 6/26 slice runs). *Grid:* `-j` ∈ {0.15, 0.2, 0.25} on the screen; `-R 3` at the
lowest stable value; and an **adaptive fallback** (run at the lower α_min, re-run only the
slices flagged by the residual QC at 0.3). *Runtime:* ≈ 25 min (screens) + 46 min
(full-plane confirmation of the choice). *Decision:* smallest α_min with 0 flagged slice runs,
plus one grid step of margin — or the adaptive fallback if the boundary is slice-specific
and the lower α_min clearly reduces lesion bias. *Caveat:* one noise realisation; optionally
confirm the boundary on a second phantom seed (`--seed 1`, 12 min to generate).

**Stage 2 — regularisation strength and type (at the stage-1 α_min).**
*2a strength, decoupled from stability:* `--l1val` scales only the wavelet threshold
(α_min also weights the coil penalty); grid {0.25, 0.5, 1, 2} → screen (stability) + lesion
slab, ≈ 1 h. *2b type:* joint l1-wavelet (current) vs total variation on the maps via ADMM
(`-r T:6:0:λ`, 3 values of λ; moba's `-r` path is experimental) vs wavelet on M_ss/M0' only
(`--not-wav-maps 1`, R1* unregularised — expected unstable, screen only), ≈ 1 h.
*Decision:* lowest lesion bias vs ideal at WM/GM bias within ±2 % and no flagged slices,
with the researcher's visual judgement from the grid figures.

**Stage 3 — iteration counts.** `-i` ∈ {8, 10, 12, 14} × `-C` ∈ {50, 100} on the lesion slab
(≈ 70 min; a subset if stages 1-2 make some cells moot) + convergence curve (relative change
vs the longest run, as in the LLR sweep). *Decision:* the cheapest setting whose lesion T1 is
within 1 % of the longest run and whose screen stays stable (more FISTA iterations
destabilised a borderline slice with the old coil model, section 7f).

**Stage 4 — coil model and initialisation.** `--sobolev_a` ∈ {110, 220, 440}, and
`ksp-sens` initialisation from `shared_selfcal(nmaps=1)` (converted by
`diag_slice.ksp_from_coils`) vs moba's default zero coils; screen + slab, ≈ 75 min.
*Decision:* the smoothest coil model that keeps WM/lesion bias at the `a = 220` level;
selfcal initialisation only if it helps on the phantom (on real data it also matters for the
coronal band artefact of single-map ENLIVE, which the phantom cannot show).

**Stage 5 — image-quality sweep for visual assessment.** A 3 × 3 grid around the chosen
point (rows: α_min or `l1val`; columns: Newton steps) on the lesion slab (≈ 80 min), figures
and phantom table per cell. Then the final full-plane run (46 min) and AX/SAG (≈ 90 min).

**Open design items (not tuning; later, on request):** 3D `moba` instead of slice-wise (the
fixed, noise-referenced scaling removes V1's reason for its failure; one 3D run ≈ 1-2 h
here and needs its own α calibration); re-estimating the per-TI phase from the model
(section 5, alternative 2); echo-group combination (joint vs mean R1*).
