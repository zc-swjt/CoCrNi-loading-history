# CoCrNi Loading-history Calculations

Computational and analysis source code for **Loading-history effects on crack-tip plasticity and stress redistribution in CoCrNi**, by Can Zhang, Mingxing Shi and Xiangyu Li.

## Scope

This repository contains the numerical methods used for the study: prescribed mixed-mode boundary loading, anisotropic crack fields, local configurational stresses, stacking-fault areas, reference-coordinate mapping, partial-dislocation motion and local virtual work. It does not contain manuscript files, figure files, plotting programs or simulation-monitoring programs.

## Code

- `engine/`: study-specific boundary-loading changes relative to GPUMD 5.5, supplied as four modified files and a patch. Replace the corresponding files under GPUMD's `src/` directory, or apply the patch with `git apply --ignore-whitespace engine/mixed_mode_boundary.patch` from the GPUMD root. Then build using its CUDA makefile. The unmodified engine is available from the [GPUMD project](https://github.com/brucefan1983/GPUMD).
- `cpu_evaluator/nep_cpu_eval.cpp`: the per-atom force and virial evaluation interface. The NEP_CPU base revision is `af615ce0e81bbb860e77923762c505567affa8d1`; the two modified neighbour-capacity source files and their patch are included. Build the interface with the `nep.cpp`, `neighbor_nep.cpp` and `ewald_nep.cpp` sources from [NEP_CPU](https://github.com/brucefan1983/NEP_CPU), using a C++17 compiler and OpenMP. The output contains atom count, energies, forces and nine virial components.
- `code/schedule.py`: the adopted loading-amplitude and angular interpolation for the 30-degree family. The 45-degree family has the same amplitude and a 15-degree shift of the loading angles. Stress analysis uses each case's actual two-component loading history.
- `code/path_theory_baseline.py`: anisotropic plane-strain crack stresses in the specimen coordinates.
- `code/dislocation_energy_basis.py`: cubic elastic tensor, anisotropic line-energy tensor and prelogarithmic energy coefficient.
- `code/material_line_map.py`: local affine inverse mapping of dislocation lines to reference coordinates, with periodic z images.
- `code/slip_plane_audit.py` and `code/track_reload_slip.py`: plane assignment, Burgers-vector classification, line sampling and correspondence.
- `code/boundary_sweep.py`: arc-length resampling, periodic line alignment and signed areas swept by partial dislocations.
- `code/fault_area.py` and `code/structure_transitions.py`: regional fault areas, atom-wise area changes and structural transitions.
- `code/virtual_glide_balance.py`: fixed-end in-plane glide mode, Peach-Koehler work, line-energy derivative and swept-area derivative.
- `code/compute_quantities.py`: stress-sector averages, two-component incremental elastic references, glide velocities and virtual-work decomposition.
- `potential/nep.txt`: the parameter file used for the simulations, with its upstream licence.
- `DATA_SOURCES.md`: physical data sources, model parameters and array definitions.

Install the Python dependencies with `python -m pip install -r requirements.txt`. The modules operate on the supplied arrays or calculation outputs; paths are passed as arguments rather than tied to the original workstation. The tests can be run with `python code/test_calculations.py`.

## Licences

The GPUMD and NEP_CPU licences are retained in `engine/LICENCE` and `cpu_evaluator/LICENSE`. The potential licence is in `potential/LICENSE`. These third-party components retain their respective licences; this repository does not replace them with a blanket licence.
