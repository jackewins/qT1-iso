# CALIPR work — handoff from the cloud session (2026-10-09)

Branch: `recon-calipr` (merged with `recon-comparison` up to phantom v2). Everything below is
committed; derived data (phantoms, reconstruction caches) lived outside the repo in the cloud
container and are **not** transferred — regenerate them locally (section 2).

## 1. Where things stand

| item | state | where |
|---|---|---|
| CALIPR V1 (single plane, 4 TIs, `pics -B`) | built, validated end to end | `Recon_CALIPR_V1.ipynb`, code factored into `synth/calipr.py` |
| Task 1: why lesion T1 is pulled towards WM | **done** | `CALIPR_LesionBias_V1.ipynb` (run on phantom v1) |
| Phantom v2 (centred truth maps, WM 250–300 / GM 310–370 ms) | **done**, pushed to `recon-comparison` | `synth/phantom.py`, `RECON_COMPARISON.md` |
| Tuning stage 1: λ × iterations (true maps, noisy, COR) | **done**, λ choice pending | `Recon_CALIPR_Tuning_V1.ipynb` |
| Tuning stages 2–6 (single plane) | planned, **paused** | `docs/calipr/D_tuning_plan.png` |
| Sequence design (which TI, how many samples, which plane) | analysis done; researcher acquired the multi-plane phantom data | `synth/ti_sampling_crlb.py`, section 4 below |
| **Multi-plane "CALIPR-SR"** (one TI per plane, joint isotropic recon) | approach chosen with literature support, feasibility check done, **not built yet** | section 5 below |

## 2. Local setup

```bash
git fetch origin && git checkout recon-calipr
# BART v1.0.00 (or later — then re-run synth/check_pics_basis_sensdim.py)
export BART_TOOLBOX_PATH=/path/to/bart
pip install numpy scipy matplotlib nbformat nbconvert jupyter
python synth/phantom.py ~/work/phantom             # phantom v2, ~8 min on 4 cores, ~280 MB
python synth/check_phantom.py ~/work/phantom
python synth/phantom.py ~/work/phantom_nf --noise 0 # noise-free twin (same seed, same phases), only if needed
```

* Notebooks read `BART_TOOLBOX_PATH` (fallback `/Users/jackewins/bart/`), `QT1_PHANTOM_DIR`
  (default `~/work/phantom`), and cache dirs `QT1_TUNE_CACHE` (default `~/work/tune`) /
  `QT1_BIAS_CACHE`.
* **Executing a notebook recomputes every missing run automatically.** Re-executing
  `Recon_CALIPR_Tuning_V1.ipynb` = the 18 stage-1 runs (~1.5 h on 4 cores in the cloud; faster
  on a 10-core laptop). The executed notebooks already contain all outputs, so only re-execute
  when needed.
* `CALIPR_LesionBias_V1.ipynb` was run on phantom v1. To reproduce it, generate v1 with the old
  generator from a worktree (it finds `acq_spec/` relative to itself):
  `git worktree add ../qT1-v1 f2840be^ && python ../qT1-v1/synth/phantom.py ~/work/phantom_v1`
  (+ `--noise 0` into `~/work/phantom_nf_v1`). Not needed for the conclusions.
* Runtimes measured in the cloud (4 cores): ENLIVE maps ~2 min/echo group; CALIPR 80 it
  ~2.5 min/echo group (+1 min phase estimate); LLR 80 it ~11.5 min/echo group.

## 3. File map (new or changed in this session)

| file | purpose |
|---|---|
| `synth/calipr.py` | CALIPR building blocks (dictionary basis, per-TI phase estimate/demod, `subspace_recon`) — unchanged V1 logic |
| `synth/run_bias_experiments.py` | cached runner: `python synth/run_bias_experiments.py <cache> <data>:<maps>:<method>:<reg> ...` (data `noisy`/`nf`; maps `E2`/`E1`/`true`; method `CAL`/`LLR`; reg `def`, `zero[N]`, `lam<λ>`, `lam<λ>i<N>`). Writes atomically; finished runs are skipped |
| `synth/bias_diagnostics.py` | closed-form fully sampled control, curve-mixing fit, blur of the ideal |
| `synth/make_lesion_bias_nb.py`, `synth/make_tuning_nb.py` | generators for the two notebooks (edit, regenerate, execute) |
| `synth/ti_sampling_crlb.py` | Cramér–Rao TI-allocation analysis (section 4) |
| `synth/check_pics_basis_sensdim.py` | proves `pics -B` ignores per-scan maps (section 5) |
| `docs/calipr/*.png` | explanatory figures: A the 12 phantom datasets, B the v1 truth-mask offset, C true vs ENLIVE maps (phase pole), D tuning plan, E stage-1 trade-off |

## 4. Findings

**Task 1 — lesion T1 pulled towards WM (coronal, phantom v1, `CALIPR_LesionBias_V1.ipynb`).**
* Pipeline itself unbiased: fully sampled, noise-free, true-map control reproduces `ideal` to ±0.3 %.
* Noise: no contribution. Undersampling itself (true maps, λ → 0): ≤ ~1–4 %.
* **Main cause: self-calibrated ENLIVE maps.** 30–57 % of each voxel's map vector unexplained vs the
  true maps; with 1 ENLIVE set 31 % of the noise-free k-space cannot be fitted (true maps 3–5 %);
  2 sets "fix" data consistency only by doubling the unknowns. ENLIVE 2-set maps show a **phase
  pole** inside the head (`docs/calipr/C_...png`), the mechanism suspected for the real-data band.
* Error shape is contrast loss at unchanged lesion width (not blur): the lesion curve reads as
  ~50 % lesion / 50 % surrounding WM. Interpretation (inferred): wrong maps → imperfect unfolding;
  identical sampling at every TI makes the leaked signal a valid IR curve, so neither LLR nor a
  subspace can reject it; the regulariser hides the artefacts.
* Regulariser (λ = 0.005, true maps): a further 2–8 points, largest for 4–6 mm lesions.
* Side findings: CSF failure is the fragile TI800 phase reference (CSF near its null there), not
  a sign flip (the grid fit allows negative M0). Unregularised CG is semi-convergent (100 vs 300
  iterations differ by 59 %).

**Phantom v2.** Truth block means were offset by (f−1)/2 fine samples (1.9 mm through-plane);
now centred. T1: WM 250–300, GM 310–370 ms (bell-shaped within tissue), lesions 357.5 / 192.5 ms.
The LLR table in `RECON_COMPARISON.md` is still v1 → regenerate locally.

**Stage 1 (CALIPR, true maps, noisy v2, COR; `Recon_CALIPR_Tuning_V1.ipynb`).**
* λ ≥ 0.005 converged by 40–80 iterations; λ ≤ 0.002 never converges (noise grows with
  iterations: the iteration count becomes the regulariser) → avoid.
* At 80 it: λ 0.005 → WM error 29 ms, mean |lesion bias| 7.5 %; λ 0.01 → 17 ms, 9.6 %;
  λ 0.02 → 11 ms, 12.1 %. Lesion bias floors at ~5 % even at tiny λ (noise in the fit; ≈ 0 noise-free).
* Recommendation: λ 0.005–0.01, 80 iterations (lean 0.005 because 3-plane combination averages
  noise but not bias). **Researcher has not chosen yet.**

**Sequence design (`synth/ti_sampling_crlb.py`).** TI150 carries the most T1 information per
unit time, then TI400; TI800 is the PSIR phase reference; TI31 anchors M0/inversion efficiency.
With ≥ 10–15 % time per TI: sample shares ~ TI31 12–18 %, TI150 53–60 %, TI400 18–22 %, TI800
7–11 % (16–20 % lower T1 SD at equal time). If inversion efficiency must be fitted, precision is
~2.4× worse and reallocation barely helps → calibrate it once on the phantom. Proposed plane
pattern ("design A"): AX TI150 | COR TI400 | SAG TI31 + TI800. **The researcher has acquired
multi-plane phantom data following this pattern** (exact protocol, R per scan and reference T1
still to be supplied).

## 5. Next: multi-plane CALIPR-SR (researcher's chosen direction)

**Literature support** (no paper combines all pieces; each piece is validated):
Van Steenkiste et al. MRM 2017;77:1818 (high-res T1 directly from low-res IR images with different
slice orientations); Beirinckx et al. CMIG 2022 (same with joint motion); Bano et al. MRM
2020;83:906 (model-based super-resolution from 10×-undersampled acquisitions); Hufnagel et al.
PMB 2024 (k-space-based model-based SR T1 from rotated stacks; better than image-space SR);
CALIPR (Dvorak et al. Sci Adv 2023) and T2 shuffling (Tamir et al. MRM 2017) for k-space subspace
reconstruction. Deoni et al. MRM 2022 (register to iso grid, fit voxelwise) is the simple baseline.

**Method.** Unknowns: K coefficient images on a 1.8 mm isotropic grid. Scan s (one TI) is modelled
as `y_s = NUFFT_s( maps · Σ_k Φ[TI_s, k] x_k )`. The scans are 3D-encoded, so the 5 mm "slice" is
just a shorter k-space extent: expressing each scan's trajectory in the common isotropic k-space
(permute axes per `geom`, scale by FOV ratio, FOV-offset phase) is an **exact** forward model.
In BART: `pics -B`, scans stacked in dim 5 (unequal sample counts via the zero-weight pattern of
`pu.stack_tis`), one basis row per scan.

**Constraints found (verified):**
* `pics -B` ignores sensitivity maps that vary along dim 5 (`synth/check_pics_basis_sensdim.py`).
  → shared maps only (fine: coils are fixed to the scanner). Per-scan phase: constant → demodulate
  k-space (as V1); linear (echo-group phase along each scan's readout) → shift that scan's
  trajectory; anything else → custom forward operator in Python (BART NUFFT + CG). LLR on the iso
  grid (`-R L`, no basis) does accept per-scan maps.
* **K = 2, not 3.** With AX TI150 / COR TI400 / SAG TI31+TI800, high L/R spatial frequencies are
  measured only by TI150 and TI400, and frequencies high along two axes by a single scan (two for
  S/I+A/P). K = 3 is underdetermined there. Use K = 2 and a stronger penalty on the T1-carrying
  coefficient (structure from all scans at full resolution, T1 contrast somewhat coarser —
  inherent to one-TI-per-orientation, for any method).

**Proposed steps (each run > 10 min needs the researcher's go-ahead):**
1. Synthetic emulation of the researcher's exact protocol: pick AX_TI150, COR_TI400, SAG_TI31,
   SAG_TI800 from the synthetic 12 scans, thinned to their samples per scan — better, add their
   real sampling coordinates as a new acquisition spec and regenerate.
2. Iso-grid truth for evaluation: `truth_iso.npz` has labels/T1 but no resolution-limited ideal
   or partial-volume masks on the iso grid → add them to `phantom.py` (shared code, on
   `recon-comparison`, with the researcher's OK).
3. Build CALIPR-SR (K = 2 and 3), coil maps on the iso grid (`ncalib` across all scans, coils
   shared; or `phantom.coil_maps` at iso coordinates as the oracle), per-scan phase as above,
   PSIR referenced to the TI800 scan, `fit_t1_grid`.
4. Baselines: Deoni-style (each scan reconstructed alone → resampled to iso → voxelwise fit) and
   LLR on the iso grid.
5. Apply to the physical-phantom data (needs: raw `.h5` or converter CFLs with TI, TR, matrix, FOV,
   orientation/offsets, sampling coordinates, echo groups; phantom reference T1 and temperature;
   any long-TI IR series for inversion efficiency).

**Paused single-plane stages** (if resumed): 2 K ∈ {2,3,4} × λ ∈ {λ*/2, λ*, 2λ*}; 3 regulariser
form (W:7 vs W:3, joint sparsity W:7:64, coefficient weighting, TV); 4 phase handling (incl.
real-valued `-c`); 5 dictionary; 6 confirm on all planes + ENLIVE maps.

## 6. Pitfalls already hit
* `pics` without `-S` returns the image in its internal scaling.
* ENLIVE 2 sets + unregularised CG breaks down (data residual 6× the data norm) — discard such runs.
* Long runs: container restarts killed background jobs twice; the runner now writes results
  atomically and skips finished conditions — just relaunch the same command.
* The noise-free phantom (`--noise 0`, same seed) has identical per-TI phases to the noisy one.

## 7. Working rules carried over
Propose every run longer than ~10 min (and every sweep) with settings, grid, runtime and the
decision each outcome leads to, and wait for the go-ahead. Use `synth/pipeline_utils.py` for
everything shared with the other methods. Report bias vs `ideal`. Commit code and executed
notebooks (phantom figures only, never data). Explain reasoning, flag uncertainty. Phantom data
only (synthetic, and the researcher's physical-phantom scans) unless the researcher says otherwise.
