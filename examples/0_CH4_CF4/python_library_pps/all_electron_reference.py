"""All-electron reference for the C 1s binding energies of CH4 and CF4 (needs PySCF).

Two final states:
  unpolarized  occupation 1 in the C 1s orbital of a spin-restricted calculation
               (half an electron removed from each spin): the all-electron analogue of
               the pseudopotential calculation, whose atoms and molecules are
               spin-unpolarized;
  polarized    one C 1s electron of one spin removed, spin-unrestricted, maximum overlap.
Scalar-relativistic (spin-free X2C), as the pseudopotential generator.  The geometries
are those of the solver inputs (CH4 inline, CF4 read from the given parsec.in).

    python all_electron_reference.py ../python_pbe/CF4/IS/parsec.in q                 # PBE
    python all_electron_reference.py ../python_pbe/CF4/IS/parsec.in q LDA_X,LDA_C_PZ  # CA

``q``: aug-cc-pwCVQZ on C, aug-cc-pVQZ on H and F (``t`` for the triple-zeta sets).
The numbers printed by these two commands are the ``AE`` table of analyze.py.
"""
import sys
import time

import numpy as np
from pyscf import dft, gto, scf

HA = 27.211386245988
XC = "PBE"
GEOM = {
    "CH4": """C -0.0000238747  -0.0000113079  -0.0000127758
H  0.9143940256  -0.5967046138   0.0932953040
H -0.8690113457  -0.6666334737  -0.0370590876
H -0.0893098756   0.6689700923   0.8633487856
H  0.0439709496   0.5943358591  -0.9196600751""",
    "CF4": None,   # read from the solver input given on the command line
}


def cf4_geometry(path):
    lines = open(path).read().splitlines()
    atoms, symbol, inside = [], None, False
    for line in lines:
        s = line.strip()
        if s.startswith("Element_Symbol:"):
            symbol = s.split(":")[1].strip()
        elif s.lower().startswith("begin atom_coord"):
            inside = True
        elif s.lower().startswith("end atom_coord"):
            inside = False
        elif inside and s:
            atoms.append(f"{symbol} {s}")
    return "\n".join(atoms)


def settle(mf):
    mf.xc = XC
    mf.grids.level = 5
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    return mf


def run(name, atoms, basis):
    t0 = time.time()
    mol = gto.M(atom=atoms, basis=basis, verbose=0, unit="Angstrom")
    ground = settle(dft.RKS(mol).x2c())
    e0 = ground.kernel()
    s = ground.get_ovlp()
    # the C 1s molecular orbital: largest weight of the first carbon s function
    carbon = [i for i, label in enumerate(mol.ao_labels()) if " C 1s" in label][0]
    weight = np.abs((s @ ground.mo_coeff)[carbon] * ground.mo_coeff[carbon])
    core = int(np.argmax(weight[: mol.nelectron // 2]))
    ref = ground.mo_coeff[:, core]

    hole = settle(dft.RKS(mol).x2c())

    def get_occ(mo_energy=None, mo_coeff=None):
        occ = np.zeros(mo_energy.size)
        occ[np.argsort(mo_energy)[: mol.nelectron // 2]] = 2.0
        occ[int(np.argmax(np.abs(ref @ s @ mo_coeff)))] = 1.0
        return occ

    hole.get_occ = get_occ
    dm = ground.make_rdm1()
    e1 = hole.kernel(dm)
    assert abs(hole.mo_occ.sum() - (mol.nelectron - 1)) < 1e-9, hole.mo_occ.sum()

    ion = gto.M(atom=atoms, basis=basis, verbose=0, unit="Angstrom", charge=1, spin=1)
    pol = settle(dft.UKS(ion).x2c())
    nocc = mol.nelectron // 2
    occ = np.zeros((2, ground.mo_coeff.shape[1]))
    occ[:, :nocc] = 1.0
    occ[0, core] = 0.0
    # pyscf convention: alpha is the majority spin of a doublet; put the hole in beta
    occ = occ[::-1].copy()
    mo0 = np.array([ground.mo_coeff, ground.mo_coeff])
    pol = scf.addons.mom_occ(pol, mo0, occ)
    e2 = pol.kernel(pol.make_rdm1(mo0, occ))
    print(f"{name}: basis functions {mol.nao}; C 1s is MO {core}, eigenvalue {ground.mo_energy[core] * HA:.3f} eV; "
          f"converged {ground.converged} {hole.converged} {pol.converged}; {time.time() - t0:.0f} s")
    print(f"{name}: E0 {e0:.8f} Ha; BE unpolarized hole {(e1 - e0) * HA:.4f} eV; "
          f"BE polarized hole {(e2 - e0) * HA:.4f} eV", flush=True)
    return (e1 - e0) * HA, (e2 - e0) * HA


if __name__ == "__main__":
    GEOM["CF4"] = cf4_geometry(sys.argv[1])
    quality = sys.argv[2] if len(sys.argv) > 2 else "q"
    if len(sys.argv) > 3:
        XC = sys.argv[3]
    print("functional", XC, "basis quality", quality)
    core_basis = {"t": "aug-cc-pwcvtz", "q": "aug-cc-pwcvqz"}[quality]
    other = {"t": "aug-cc-pvtz", "q": "aug-cc-pvqz"}[quality]
    out = {}
    for name in ("CH4", "CF4"):
        out[name] = run(name, GEOM[name], {"C": core_basis, "H": other, "F": other})
    print(f"shift CF4 - CH4: unpolarized {out['CF4'][0] - out['CH4'][0]:.4f} eV; "
          f"polarized {out['CF4'][1] - out['CH4'][1]:.4f} eV")
