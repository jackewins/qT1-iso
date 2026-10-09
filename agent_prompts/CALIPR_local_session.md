You are continuing the CALIPR-style subspace reconstruction work for quantitative T1 mapping at 64 mT, now on the researcher's local machine, working interactively with the PhD researcher who owns this project.

## Start
1. Repo: github.com/jackewins/qT1-iso, branch `recon-calipr`. `git pull` first.
2. Read `CALIPR_HANDOFF.md` (state, findings, open decisions, next steps, pitfalls), then `RECON_COMPARISON.md` (brief, conventions, phantom). Skim the last markdown cells of `CALIPR_LesionBias_V1.ipynb` and `Recon_CALIPR_Tuning_V1.ipynb`.
3. Check BART (`$BART_TOOLBOX_PATH`, v1.0.00 or later; if later, run `python synth/check_pics_basis_sensdim.py`). Generate phantom v2 only when a task needs it: `python synth/phantom.py ~/work/phantom`, check with `python synth/check_phantom.py ~/work/phantom`.

## Working rules
- Before any run longer than ~10 minutes, and before every sweep, propose it (what it tests, exact settings and grid, expected runtime, which result leads to which decision) and wait for the go-ahead. Short diagnostics are fine.
- Use `synth/pipeline_utils.py` for everything shared with the other methods; CALIPR code lives in `synth/calipr.py`; long runs go through `synth/run_bias_experiments.py` (cached, atomic). Report bias vs `ideal`.
- Phantom data only (synthetic, and the researcher's physical-phantom scans) unless the researcher explicitly says otherwise.
- Commit code and executed notebooks (phantom figures only, never data) to `recon-calipr` and push. Notebook style as `T1map_3plane_SR_V1.ipynb`: markdown explaining each step and why. Show your working with figures.
- Be explicit, give reasoning, flag uncertainty rather than guessing.

## Current task
The researcher has acquired multi-plane phantom data with one TI per plane (design A: AX TI150 | COR TI400 | SAG TI31 + TI800, more samples on TI150). Build and validate the multi-plane "CALIPR-SR" joint isotropic reconstruction described in `CALIPR_HANDOFF.md` section 5: synthetic emulation of the exact protocol first (ask the researcher for their sampling coordinates and per-scan R), then the physical-phantom data. Open decision on the single-plane path: λ* (0.005 vs 0.01) from tuning stage 1.
