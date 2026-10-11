# Exact Gaussian–Hermite closure of the mapping quantum–classical Liouville equation

Code, configuration and archived results for the article *Exact
Gaussian–Hermite closure of the mapping quantum–classical Liouville equation*
by Sahand Nikzat, Ali Nassimi, and Rony Modak Parai.

In the Meyer–Miller–Stock–Thoss mapping representation, the Liouville operator
of the quantum–classical Liouville equation (QCLE) contains an excess coupling
term besides the Poisson bracket; discarding it gives the Poisson bracket
mapping equation (PBME).  Expanded in Hermite functions about the Gaussian
shared by all physical mapping densities, the exact dynamics never raises the
Hermite order and closes on a finite, sparse set of equations for coefficient
fields.  This repository propagates those equations for Tully's dual avoided
crossing, checks them against independent solvers of the subsystem QCLE and
against converged quantum dynamics, compares them with the PBME, and contains
the computer-algebra scripts behind every symbolic statement of the article.
Every number in the article is a macro written by `scripts/15_paper_values.py`
from the archived outputs in `results/raw`.

## Layout

| Path | Contents |
|---|---|
| `src/mqcle/` | the package: models, mapping, Hermite algebra (`hermite/`), coefficient operator and propagators, PBME, grid QCLE and Schrödinger references |
| `configs/` | model and global parameters (`base.yaml`, `tully_default.yaml`); the production settings of the six momentum regimes are in `src/mqcle/benchmark_regimes.py` and on the runner's command lines |
| `scripts/` | one script per stage, numbered in pipeline order (table below) |
| `tests/` | 255 tests, including the Gaussian–Hermite gates GH-01 to GH-15 |
| `run_final_pipeline.ps1` | the production runner (Windows PowerShell) |
| `results/raw/` | JSON output of every stage and the HDF5 archives up to 50 MiB |
| `results/failed/P10_2048x1024_20260829/` | the coarse-grid P0 = 10 reference run used by the two-grid acceptance of `07c_p10_refinement_acceptance.py` |
| `results/logs/` | stage logs |
| `results/figures/paper_*` | figures of the article and its supplement |
| `results/tables/paper_values.json` | every quoted number with the file and quantity it comes from |
| `results/PROVENANCE.md` | command line, platform, Python version, time and wall time of every archived output |
| `manuscript/paper_values.tex` | the LaTeX macros the article reads its numbers from |
| `MANIFEST.sha256` | SHA-256 digest of every file of the project, including the archives kept out of Git |

## Stages

| Script | Stage |
|---|---|
| `00`–`05`, `17` | validation of the original-space physics and exact computer-algebra checks: polynomial and Hermite closure, QCLE correspondence, physical frame, further cases, generator structure |
| `07g` | **Gaussian–Hermite coefficient propagation** (the method of the paper) |
| `07cg`, `07k`, `09g`, `10g` | its basis covariance, closure validation, convergence and cross-check |
| `07`, `07b`, `07c`, `09` | subsystem-QCLE grid reference (Strang splitting of ρ_W) with exact quantum dynamics and focused PBME in the same run; grid independence; basis covariance; convergence |
| `07c_p10_refinement_acceptance` | two-grid acceptance of the P0 = 10 reference |
| `10`, `10b` | convergence of the quantum reference; independent four-field grid solver of the QCLE |
| `08` | fourth-order defect rate (Hermite leakage) per regime, and the six-regime and coupling-scan aggregates (`--aggregate`) |
| `11`, `11b`, `11c`, `18`, `19`, `20` | projected-density PBME, its sampling and time-step convergence, phase-space densities and focused-PBME observables |
| `15`, `16`, `21` | paper values, paper figures, provenance table |
| `13`, `14` | acceptance audit and journal-format audit of the manuscript |

The outputs of `07_coefficient_propagation.py` are the subsystem-QCLE
reference, not Gaussian–Hermite output; the two branches are labelled
separately throughout.

## Requirements

Python 3.11 or later with the packages in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

The LaTeX build of the manuscript (stages Postprocess and Audit of the runner)
needs a TeX distribution with REVTeX 4.2.

## Reproducing the results

Run the tests:

```
python -m pytest tests -q
```

Regenerate the paper values and figures from the archived outputs:

```
python scripts/15_paper_values.py
python scripts/16_paper_figures.py
python scripts/21_provenance_table.py
```

Scripts 15 and 16 also read the field archives listed below, which are not in
the repository; place them in `results/raw/` first.

Rerun the calculations with the runner.  It prints its plan by default, and
every stage is skipped when its validated output already exists, so a stage is
recomputed by moving its output out of `results/raw/`:

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_final_pipeline.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_final_pipeline.ps1 -Phase Heavy
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_final_pipeline.ps1 -Phase Hermite
```

The production runs take days on one workstation; Table I of the article lists
the wall times of the coefficient propagation.  Stages that the runner does not
start record the command that produced them, and `results/PROVENANCE.md` lists
the command of every archived output.  The Postprocess and Audit phases build
and check the manuscript, whose sources are not part of this repository.

## Files not in the repository

The following HDF5 field archives exceed GitHub's 50 MiB recommendation
(3.08 GB in total).  They are available from the corresponding author on
reasonable request; their SHA-256 digests are in `MANIFEST.sha256`.

| File | Size |
|---|---|
| `results/raw/07g_gh_coefficient_propagation_P10.h5` | 957.9 MB |
| `results/raw/07g_gh_coefficient_propagation_P20.h5` | 496.4 MB |
| `results/raw/07g_gh_coefficient_propagation_P30.h5` | 63.5 MB |
| `results/raw/07g_gh_coefficient_propagation_P40.h5` | 63.1 MB |
| `results/raw/07g_gh_coefficient_propagation_P70.h5` | 250.1 MB |
| `results/raw/07g_gh_coefficient_propagation_P100.h5` | 246.1 MB |
| `results/raw/10b_qcle_grid_crosscheck_P10.h5` | 201.6 MB |
| `results/raw/10b_qcle_grid_crosscheck_P20.h5` | 402.9 MB |
| `results/raw/10b_qcle_grid_crosscheck_P70.h5` | 201.5 MB |
| `results/raw/10b_qcle_grid_crosscheck_P100.h5` | 201.4 MB |

Code comments refer to development records (`CORRECTIONS.md`,
`CLAIM_TABLE.md`, `MODULE_STATUS.md`) that are likewise kept outside the
repository; `scripts/13_acceptance.py` reads them.

## Checking file integrity

Git stores every file byte for byte (`.gitattributes`), so the digests hold in
a clone:

```
sha256sum -c --ignore-missing MANIFEST.sha256
```

## License

MIT; see `LICENSE`.  If you use this code, please cite the article.
