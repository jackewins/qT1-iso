# moba_dev — helper scripts from the first moba build (2026-09-29)

Working scripts the first agent used to build and run `Recon_MOBA_V1.ipynb`, saved so the
next session can continue from them. They were written for a local run:

* `REPO` / paths at the top of `moba_core.py`, `t2d.py`, `quick_eval.py`, `queue.sh` point to
  a local worktree — change them to your checkout.
* Phantom data were in `~/work/moba/phantom` (regenerate with `python synth/phantom.py <dir>`).

Files: `moba_core.py` (core moba call and helpers), `t2d.py` (readout iFFT to 2D slices),
`run_partial.py` / `queue.sh` (slice-parallel runs), `quick_eval.py` (evaluation),
`build_nb.py` / `assemble.py` (notebook generation / assembly).
