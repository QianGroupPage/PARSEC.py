#!/usr/bin/env python3
"""Build a clean pbe/ + b3lyp/ tree reproducing the CH4 / CF4 rows of Table S1.

Reference: Xu, Prendergast, Qian, J. Chem. Theory Comput. 2022, 18, 5471-5478.
The paper's own production runs survive on CFS; set ARES_DSCF_DATA to their
DSCF_DATA/C folder and ARES_PP_DIR to their PPs_copy pseudopotentials:

    $ARES_DSCF_DATA/{2,15}/{I,F}

(2 = CH4, 15 = CF4; I = initial state, F = 1s final state).  Everything below
-- geometry, pseudopotentials, grid spacing, cutoff radius -- is taken from
those directories rather than reconstructed, so any disagreement in the answer
is a real difference and not an input mismatch.

Two tests, differing only in one flag:

  pbe/    Lrefine=F  -> ares.log carries '*AE energy (test)'  (PP-PBE)
  b3lyp/  Lrefine=T  -> ARES additionally re-evaluates eq 4 non-self-consistently
                        with the B3LYP hybrid on the converged PBE orbitals and
                        density, printing '*ref.AE energy'    (PP-PBE(B3LYP))

The B3LYP refinement is a post-processing step on the PBE solution, so a
Lrefine=T run reports both columns.  Running the Lrefine=F test alongside it
confirms that turning the refinement on leaves the underlying SCF untouched.

Two details worth flagging, both differing from our earlier archived attempt:

* The paper does NOT use a uniform grid.  CH4 ran at GRIDSPACING=0.12 /
  ISOrmax=5.0 (84^3 grid) and CF4 at 0.07 / 5.5 (158^3).  Reproducing Table S1
  means reproducing those per-molecule choices, not a single global setting.

* The core-hole carbon stays labelled 'C' in POSCAR; only the pseudopotential
  changes (C.pbe-mt-cpi-1s.UPF) and DCHARGE goes to -1.0.  These molecules have
  a single carbon, so no second species slot is needed.
"""
import os
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REF = os.environ.get("ARES_DSCF_DATA", "")    # DSCF_DATA/C of the JCTC 2022 runs
PPDIR = os.environ.get("ARES_PP_DIR", "")     # their PPs_copy pseudopotentials


def require_inputs():
    """Stop with a message unless both reference folders are named and exist."""
    for name, value, what in (
            ("ARES_DSCF_DATA", REF, "the DSCF_DATA/C folder of the JCTC 2022 runs (2/I, 2/F, 15/I, 15/F)"),
            ("ARES_PP_DIR", PPDIR, "the PPs_copy pseudopotentials of those runs (C.pbe-mt-cpi.UPF, ...)")):
        if not value:
            sys.exit(f"setup.py: set {name} to {what}")
        if not os.path.isdir(value):
            sys.exit(f"setup.py: {name}={value} is not a folder; it must be {what}")

# molecule -> (reference index, ligand, GRIDSPACING, ISOrmax)
MOL = {
    "CH4": dict(idx=2,  ligand="H", gridspacing="0.12", isormax="5.0"),
    "CF4": dict(idx=15, ligand="F", gridspacing="0.07", isormax="5.5"),
}

# Verbatim from DSCF_DATA/C/2/I/ares.in, with the four run-dependent fields
# templated out.  Comment lines are the paper's own.
ARES_IN = """\
#########################BASIS DATA##############################
Lrefine={lrefine}
SYSTEM = {system}
#Cell file name
CELLFILE = POSCAR
#real-space gridsize (angs)
GRIDSPACING = {gridspacing}
#pseudo-potential
PPFILE = {ppfile}
#Finite difference order
NORDER = 8
#Switch for spin
LSPIN= T
#Spin polarized
Snlcc=0.5
#The order of smearing (2-5) , <=0 is fermi-dirac-like smearing
Nsmear= -1
#Width of broadening(eV.)
Wsmear= 0.1
#Number of simulate steps
Nssp= 0
#ISTART 0: SCF/MD/RELAX 1: NonSCF
ISTART= 0
#Output data
LWAVE=F
LCHARGE=F
#LMOM
LMOM=F
##############################Diag################################
#0:ARPACK 1:Chebyshev
Idiag=1
#Chebyshev order [8,20]
CheM=12
CheM0=-24
#For add eigenstates to diag(H)
NADST= 16
#first step use diag(H)
Lfirst=F
#RayleighRitz need OrthNorm?
LOrthNorm=F
#LINRHO
LINRHO=F
######################For Isolated systems########################
#The max sphere radius
ISOrmax = {isormax}
#Lmax>=0 Multipoles
LMAX=9
#The order of multi-grids
NVC=6
#For relative tolerance of CG iteration
TOLCG=1e-6
#D-charge
DCHARGE={dcharge}
#Linear Scale Calculations
LELE=F
##########################mixing data#############################
#mixer we used (0:rPulay,1,sAnderson,2:rPulayk)
IMIXER=2
#the max iter step
NMITER=100
#number of simple mixing
NSMIX=0
#simple mixing factor
MALPHA=0.40
#non-simple mixing factor
MBETA=0.30
#number of history for Anderson mixing used (2-6)
NHMAX= 4
#For mininal history for restart Anderson
NHMIN= 2
#for ill matrix
W0AM=0.0
#tolarence of drho in scf
RTOL=1e-5
#tolerance of total energy(Bohr)
ETOL=5e-6
############################OUT-PUT###############################
############################THE END###############################
"""


def build(level, lrefine):
    for mol, cfg in MOL.items():
        for state, refstate in (("IS", "I"), ("FS_1s", "F")):
            d = os.path.join(BASE, level, mol, state)
            os.makedirs(d, exist_ok=True)

            # Geometry straight from the paper's run directory.  The IS and FS
            # POSCARs there are identical apart from a stray zero in one cell
            # vector; ARES ignores the cell for isolated systems (the domain is
            # the ISOrmax sphere about the origin), so take the IS copy for both.
            shutil.copy(os.path.join(REF, str(cfg["idx"]), "I", "POSCAR"),
                        os.path.join(d, "POSCAR"))

            # PPFILE order must follow the POSCAR species order, which is C first.
            cpp = "C.pbe-mt-cpi-1s.UPF" if state == "FS_1s" else "C.pbe-mt-cpi.UPF"
            ligpp = f"{cfg['ligand']}.pbe-mt-cpi.UPF"
            for src in (cpp, ligpp):
                shutil.copy(os.path.join(PPDIR, src), os.path.join(d, src))

            with open(os.path.join(d, "ares.in"), "w") as f:
                f.write(ARES_IN.format(
                    lrefine=lrefine,
                    system=f"{mol}_{state}",
                    gridspacing=cfg["gridspacing"],
                    isormax=cfg["isormax"],
                    ppfile=f"{cpp} {ligpp}",
                    dcharge="-1.0" if state == "FS_1s" else "0.0"))

            print(f"built {level}/{mol}/{state}"
                  f"  (h={cfg['gridspacing']}, rmax={cfg['isormax']})")


if __name__ == "__main__":
    require_inputs()
    for level, lrefine in (("pbe", "F"), ("b3lyp", "T")):
        build(level, lrefine)
