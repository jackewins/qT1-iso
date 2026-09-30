# moba_dev — helper scripts for the moba reconstruction

Paths are repo-relative; the phantom directory comes from `QT1_PHANTOM_DIR` (default
`~/work/phantom`, generate with `python synth/phantom.py <dir>`) and BART from
`BART_TOOLBOX_PATH` (default `~/bart`). Derived data go to `<phantom dir>/moba/` (outside the repo).

## V2 (2026-09-30, first stable version)

* `diag_slice.py` — single-readout-slice harness used for the stability diagnosis
  (`Recon_MOBA_V2.ipynb` section 7) and imported by the V2 notebook: runs one readout slice
  through `moba` exactly as the full reconstruction does, reports the raw-data residual vs the
  noise floor, collapsed voxels and T1 vs ideal, caches every run in `<phantom dir>/moba/diag/cache`;
  also moba's coil-model transform (`coils_from_ksp`, verified against moba's own output;
  `ksp_from_coils` builds a `--other ksp-sens` initial value from image-space maps).
  CLI: `python moba_dev/diag_slice.py COR 56 "<moba opts>" [echo]`.
* `build_nb_v2.py` — builds `Recon_MOBA_V2.ipynb` from V1 (verified V1 cells are copied
  verbatim) plus the new sections; `report_v2.md` / `tuning_plan_v2.md` hold the report cell.
  Run from the repo root: `python moba_dev/build_nb_v2.py`.

## V1 (2026-09-29, first build) — kept for the record

`moba_core.py` (prototype of V1's moba call, 3D), `t2d.py` (V1 readout-slice test bed;
superseded by `diag_slice.py`), `run_partial.py` (execute the first N cells of a notebook into
a scratch copy), `queue.sh` (V1's job queue), `quick_eval.py` (evaluation of cached V1 runs),
`build_nb.py` / `assemble.py` (V1 notebook generation / assembly).
