#!/usr/bin/env python3
"""C 1s dSCF binding energies for CH4 / CF4: our runs vs JCTC 2022 vs experiment.

  BE    = AE(FS_1s) - AE(IS)
  dBE   = BE(CF4) - BE(CH4)

The AE (all-electron) energy of eq 4 is the right quantity rather than the
pseudo total energy, because the initial and final states use pseudopotentials
with different z_valence and their pseudo total energies are not on a common
scale.  ARES prints two of them:

  '*AE energy (test)'  PP-PBE          -- always
  '*ref.AE energy'     PP-PBE(B3LYP)   -- only when Lrefine=T

Absolute BEs carry a ~ +6 eV offset because ARES evaluates eq 4 with
spin-unpolarized atomic and ionic energies.  That offset is a per-element
constant, so it cancels exactly in dBE -- which is why dBE, not the absolute
BE, is the number to compare against experiment.

Reference columns are read live from the production runs behind Xu, Prendergast,
Qian, J. Chem. Theory Comput. 2022, 18, 5471 -- our own JCTC paper -- which still
live on CFS (set ARES_DSCF_DATA to their DSCF_DATA/C folder).  So 'JCTC 2022'
below is that paper's actual ARES output, not a transcription of its Table S1.
"""
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REF = os.environ.get("ARES_DSCF_DATA", "")    # DSCF_DATA/C of the JCTC 2022 runs
REF_IDX = {"CH4": 2, "CF4": 15}
MOLS = ("CH4", "CF4")

# Gas-phase experimental C 1s binding energies (eV), as tabulated in Table S1.
EXP = {"CH4": 290.80, "CF4": 301.85}
DBE_EXP = EXP["CF4"] - EXP["CH4"]

# Experimental gas-phase XPS is good to roughly this much.  It is the floor on
# what any calculation can be said to resolve, but it is NOT a pass/fail line --
# a shift of ~11 eV predicted to a few tenths of an eV is a good result whether
# or not it clears 0.1 eV.  Reported alongside the relative error and the
# JCTC 2022's own published mean absolute error for carbon shifts, for context.
EXP_UNC = 0.1
PAPER_MAE = 0.20   # JCTC 2022 published MAE for C 1s shifts, PP-PBE(B3LYP)

METHODS = ("PP-PBE", "PP-PBE(B3LYP)")
PAT = {
    "PP-PBE": r"\*AE energy \(test\)->\s*(-?[\d.]+)",
    "PP-PBE(B3LYP)": r"\*ref\.AE energy\s*->\s*(-?[\d.]+)",
}


def energies(path):
    """{method: AE energy} from one ares.log; {} if it did not finish."""
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return {}
    if "Well Done" not in text:
        return {}
    out = {}
    for meth, pat in PAT.items():
        m = re.findall(pat, text)
        if m:
            out[meth] = float(m[-1])
    return out


def binding_energies(logs):
    """{method: {mol: BE}} given {mol: {state: logpath}}."""
    be = {}
    for mol, states in logs.items():
        e = {s: energies(p) for s, p in states.items()}
        if not (e["IS"] and e["FS_1s"]):
            missing = [s for s in ("IS", "FS_1s") if not e[s]]
            print(f"  {mol}: unfinished ({', '.join(missing)})", file=sys.stderr)
            continue
        for meth in METHODS:
            if meth in e["IS"] and meth in e["FS_1s"]:
                be.setdefault(meth, {})[mol] = e["FS_1s"][meth] - e["IS"][meth]
    return be


def ours(level):
    return binding_energies({
        mol: {s: os.path.join(BASE, level, mol, s, "ares.log")
              for s in ("IS", "FS_1s")}
        for mol in MOLS})


def jctc2022():
    return binding_energies({
        mol: {ours_: os.path.join(REF, str(REF_IDX[mol]), theirs, "ares.log")
              for ours_, theirs in (("IS", "I"), ("FS_1s", "F"))}
        for mol in MOLS})


def main():
    if not REF:
        sys.exit("analyze.py: set ARES_DSCF_DATA to the DSCF_DATA/C folder of the JCTC 2022 runs "
                 "(2/I, 2/F, 15/I, 15/F)")
    if not os.path.isdir(REF):
        sys.exit(f"analyze.py: ARES_DSCF_DATA={REF} is not a folder")
    src = {"pbe (Lrefine=F)": ours("pbe"),
           "b3lyp (Lrefine=T)": ours("b3lyp"),
           "JCTC 2022 (published)": jctc2022()}

    print("=" * 78)
    print("absolute C 1s binding energies, eV   "
          "(BE = AE[FS_1s] - AE[IS]; ~+6 eV offset expected, see docstring)")
    print("=" * 78)
    print(f"{'run':22s} {'method':16s} {'BE(CH4)':>10s} {'BE(CF4)':>10s} {'dBE':>10s}")
    print("-" * 78)
    for name, be in src.items():
        for meth in METHODS:
            v = be.get(meth, {})
            if len(v) < 2:
                continue
            print(f"{name:22s} {meth:16s} {v['CH4']:10.4f} {v['CF4']:10.4f} "
                  f"{v['CF4'] - v['CH4']:10.4f}")

    # --- reproduction check: our numbers against JCTC 2022's own output ------
    ref = src["JCTC 2022 (published)"]
    print()
    print("=" * 78)
    print("reproduction of our JCTC 2022 runs (same geometry, PPs, grid, rmax)")
    print("=" * 78)
    print(f"{'run':22s} {'method':16s} {'d BE(CH4)':>11s} {'d BE(CF4)':>11s} {'d dBE':>10s}")
    print("-" * 78)
    for name in ("pbe (Lrefine=F)", "b3lyp (Lrefine=T)"):
        for meth in METHODS:
            v, r = src[name].get(meth, {}), ref.get(meth, {})
            if len(v) < 2 or len(r) < 2:
                continue
            print(f"{name:22s} {meth:16s} "
                  f"{v['CH4'] - r['CH4']:+11.4f} {v['CF4'] - r['CF4']:+11.4f} "
                  f"{(v['CF4'] - v['CH4']) - (r['CF4'] - r['CH4']):+10.4f}")

    # --- the physical comparison: dBE against experiment ---------------------
    print()
    print("=" * 78)
    print("chemical shift  dBE = BE(CF4) - BE(CH4)  vs experiment")
    print("=" * 78)
    # PP-PBE from the dedicated Lrefine=F test, PP-PBE(B3LYP) from the Lrefine=T
    # test.  JCTC 2022's own values are covered by the reproduction table above.
    print(f"{'method':16s} {'dBE':>9s} {'exp_dBE':>9s} {'error':>9s}")
    print("-" * 78)
    for meth, level in (("PP-PBE", "pbe (Lrefine=F)"),
                        ("PP-PBE(B3LYP)", "b3lyp (Lrefine=T)")):
        v = src[level].get(meth, {})
        if len(v) < 2:
            continue
        d = v["CF4"] - v["CH4"]
        print(f"{meth:16s} {d:9.3f} {DBE_EXP:9.3f} {d - DBE_EXP:+9.3f}")
    print()
    print(f"for scale: gas-phase XPS resolves ~{EXP_UNC:.1f} eV; "
          f"our JCTC 2022 published C-shift MAE is {PAPER_MAE:.2f} eV")

    # --- what the hybrid refinement actually buys ----------------------------
    b = src["b3lyp (Lrefine=T)"]
    if all(len(b.get(m, {})) == 2 for m in METHODS):
        d0 = b["PP-PBE"]["CF4"] - b["PP-PBE"]["CH4"]
        d1 = b["PP-PBE(B3LYP)"]["CF4"] - b["PP-PBE(B3LYP)"]["CH4"]
        print()
        print("=" * 78)
        e0, e1 = d0 - DBE_EXP, d1 - DBE_EXP
        print(f"B3LYP refinement moves dBE {d0:.3f} -> {d1:.3f} eV ({d1 - d0:+.3f})")
        print(f"  error vs experiment: {e0:+.3f} -> {e1:+.3f} eV"
              f"   ({100 * (1 - abs(e1) / abs(e0)):.0f}% reduction)")

    # --- cross-check that Lrefine=T does not disturb the SCF -----------------
    p, bb = src["pbe (Lrefine=F)"].get("PP-PBE", {}), b.get("PP-PBE", {})
    if len(p) == 2 and len(bb) == 2:
        worst = max(abs(p[m] - bb[m]) for m in MOLS)
        print()
        print(f"Lrefine=F vs Lrefine=T, PP-PBE column: max |difference| = "
              f"{worst:.2e} eV  ({'consistent' if worst < 1e-3 else 'INCONSISTENT'})")


if __name__ == "__main__":
    main()
