# FHI98PP inputs of the C, H and F pseudopotentials of the benchmark

The ARES benchmark in `../0_ARES_CH4_CF4` uses the pseudopotentials of Xu,
Prendergast, Qian, J. Chem. Theory Comput. **18**, 5471 (2022). The four files
here are FHI98PP inputs that regenerate those potentials; they were recovered
from the UPF files of the benchmark, which do not record their inputs.

| file | potential | channels | cutoff radii (bohr) | local |
|---|---|---|---|---|
| `H.fhi98pp.in` | H | s | 1.13 | s |
| `C.fhi98pp.in` | C | s, p | s 1.47, p 1.50 | p |
| `C-1s.fhi98pp.in` | C with one 1s electron removed | s, p | s 1.47, p 1.50 | p |
| `F.fhi98pp.in` | F | s, p | 1.28, 1.28 | p |

All are PBE, Troullier–Martins, scalar-relativistic, without core correction.
The carbon potential and its core-hole partner have the same radii. A radius
in the table is any two-decimal value that selects the mesh point of the
published potential (C s: mesh point 1.4620, C p: 1.4982, H: 1.1298,
F: 1.2748). The inputs published with the paper
(`../../0d_xps_core_levels/jctc2022_1s/paper/fhi98pp_inputs`: C 1.49 and 1.52,
H 1.15, F 1.29) select the same mesh points.

Regenerated from these inputs (with a Python port of FHI98PP that is not part
of this repository) and compared with the UPF files of the benchmark
(`../0_ARES_CH4_CF4/test_CH4_CF4/pbe/*/*/*.UPF`):

| potential | all-electron / pseudo-atom energy (Ha) | header of the published UPF (Ry) | largest relative deviation of a numerical block |
|---|---|---|---|
| H | −0.458933 / −0.458596 | −0.91786 / −0.91720 | 3e-13 |
| C | −37.764780 / −5.363285 | −75.52956 / −10.72658 | 9e-12 |
| C-1s | −26.579090 / −8.844287 | −53.15818 / −17.68858 | 2e-11 |
| F | −99.744308 / −23.981793 | −199.48862 / −47.96358 | 3e-11 |

Every numerical block of the UPF (mesh, local potential, semilocal potentials,
projectors, D matrix, pseudo wave functions, atomic density) agrees. The
headers differ only in notation (`core_correction="false"` for `"F"`) and in
the energies, which the published files give as twice the five-decimal Hartree
value of the FHI98PP log.
