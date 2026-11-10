"""Tables of this example from the logs: energies, gaps, forces and timings against Quantum ESPRESSO 7.5.

From the repository root::

    python examples/3d_hybrid_functionals/collect_results.py            # the reference logs parsec_python.out
    python examples/3d_hybrid_functionals/collect_results.py --logs DIR [--runs DIR/runs.jsonl]

``--logs DIR`` reads ``DIR/<case>/parsec.out`` (a directory of ``run_examples.py --out DIR``) instead, ``--runs``
the JSON lines of the runner (peak device memory).  The QE numbers below are those of the inputs in ``qe/``
(QE 7.5, the same FHI98PP tables as UPF, see ``qe/README.md``); ``QE_MOVED`` are the same QE inputs with every
atom moved off QE's FFT grid (``qe/*_moved.in``, README "QE's HSE energies and its grid").
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RY_EV = 13.605693122994

# QE 7.5 rows: total energy (Ry), Fock energy a_x E_x (Ry, "+ Fock energy (ACE)"), gap over the mesh (eV), force
# on atom 2 (Ry/bohr) of the displaced cell.
QE = {
    "si_pbe_k222_e60": (-15.57477255, None, 0.6105, None),
    "si_pbe0_k222_e60_gbx": (-15.59118868, -1.09610497, 2.2893, None),
    "si_hse06_k222_e60_gb": (-15.5949562, -0.86447055, 1.6231, None),
    "si_hf_k222_e60_gbx": (-15.07439749, -4.46160434, 7.9362, None),
    "si_b3lyp_libxc_k222_e60_gbx": (-15.5365091, -0.88257822, 2.2131, None),
    "si_pbe_k444_e60": (-15.74854796, None, 0.7006, None),
    "si_pbe0_k444_e60_gbx": (-15.75913239, -1.07070707, 1.9817, None),
    "si_hse06_k444_e60_gb": (-15.75599559, -0.83294089, 1.3274, None),
    "si_hf_k444_e60_gbx": (-15.21710704, -4.36746762, 6.7861, None),
    "si_disp_pbe_k222_e60": (-15.57432061, None, 0.4113, (-0.01806913, 0.0, 0.0)),
    "si_disp_pbe0_k222_e60_gbx": (-15.59071369, -1.0960799, 2.0801, (-0.01898746, 0.0, 0.0)),
    "si_disp_hse06_k222_e60_gb": (-15.59448117, -0.86444827, 1.4149, (-0.01889558, 0.0, 0.0)),
    "c_pbe_k333_e120": (-22.75901823, None, 4.1749, None),
    "c_pbe0_k333_e120_gbx": (-22.76126647, -1.60903339, 6.2152, None),
    "c_hse06_k333_e120_gb": (-22.76345019, -1.36981661, 5.551, None),
    "mgo_pbe_k333_e150": (-35.13885358, None, 4.6579, None),
    "mgo_pbe0_k333_e150_gbx": (-34.86879976, -1.89795571, 7.2118, None),
    "mgo_hse06_k333_e150_gb": (-34.91008875, -1.65476926, 6.4986, None),
    "hbn_pbe_k661_e100": (-25.80737855, None, 4.6973, None),
    "hbn_pbe0_k661_e100_gbx": (-25.79687397, -1.68611913, 6.3753, None),
    "hbn_hse06_k661_e100_gb": (-25.79627181, -1.44206026, 5.7089, None),
    "o2_pbe_k111_e150": (-63.71072842, None, 2.4864, None),
    "o2_pbe0_k111_e150_gbx": (-63.67028508, -3.16297699, 6.1969, None),
    "o2_hse06_k111_e150_gb": (-63.67360858, -2.80020128, 5.4187, None),
}

# The same rows with every atom moved off QE's FFT grid (qe/*_moved.in): total energy, Fock energy (Ry).
QE_MOVED = {
    "si_pbe_k222_e60": (-15.57477219, None),
    "si_pbe0_k222_e60_gbx": (-15.59118805, -1.09610496),
    "si_hse06_k222_e60_gb": (-15.5949533, -0.86447081),
    "si_pbe_k444_e60": (-15.74854757, None),
    "si_pbe0_k444_e60_gbx": (-15.75913194, -1.07070698),
    "si_hse06_k444_e60_gb": (-15.75599151, -0.8329415),
    "c_pbe_k333_e120": (-22.75901773, None),
    "c_pbe0_k333_e120_gbx": (-22.76126429, -1.60903333),
    "c_hse06_k333_e120_gb": (-22.76342768, -1.36981662),
    "mgo_pbe_k333_e150": (-35.13885377, None),
    "mgo_pbe0_k333_e150_gbx": (-34.86880028, -1.89795566),
    "mgo_hse06_k333_e150_gb": (-34.91003581, -1.65476929),
    "hbn_pbe_k661_e100": (-25.80737613, None),
    "hbn_pbe0_k661_e100_gbx": (-25.79687242, -1.68611887),
    "hbn_hse06_k661_e100_gb": (-25.79626094, -1.44206011),
    "o2_pbe_k111_e150": (-63.71074166, None),
    "o2_pbe0_k111_e150_gbx": (-63.67028562, -3.16297971),
    "o2_hse06_k111_e150_gb": (-63.67330643, -2.80019512),
}

# case -> (QE row, QE row of PBE, setting of the q+G = 0 term / slab kernel in words)
CASES = {
    "Si_k222/pbe0": ("si_pbe0_k222_e60_gbx", "si_pbe_k222_e60", "default (= QE gbx)"),
    "Si_k222/hse06": ("si_hse06_k222_e60_gb", "si_pbe_k222_e60", "gb, a = 10/ecutwfc"),
    "Si_k222/hse06_default": ("si_hse06_k222_e60_gb", "si_pbe_k222_e60", "default (a -> 0)"),
    "Si_k222/hf": ("si_hf_k222_e60_gbx", "si_pbe_k222_e60", "default (= QE gbx)"),
    "Si_k222/b3lyp": ("si_b3lyp_libxc_k222_e60_gbx", "si_pbe_k222_e60", "default (= QE gbx)"),
    "Si_k444/pbe0": ("si_pbe0_k444_e60_gbx", "si_pbe_k444_e60", "default (= QE gbx)"),
    "Si_k444/hse06": ("si_hse06_k444_e60_gb", "si_pbe_k444_e60", "gb, a = 10/ecutwfc"),
    "Si_k444/hse06_default": ("si_hse06_k444_e60_gb", "si_pbe_k444_e60", "default (a -> 0)"),
    "Si_k444/hf": ("si_hf_k444_e60_gbx", "si_pbe_k444_e60", "default (= QE gbx)"),
    "Si_disp_k222/pbe0": ("si_disp_pbe0_k222_e60_gbx", "si_disp_pbe_k222_e60", "default (= QE gbx)"),
    "Si_disp_k222/hse06": ("si_disp_hse06_k222_e60_gb", "si_disp_pbe_k222_e60", "gb, a = 10/ecutwfc"),
    "C_k333/pbe0": ("c_pbe0_k333_e120_gbx", "c_pbe_k333_e120", "default (= QE gbx)"),
    "C_k333/hse06": ("c_hse06_k333_e120_gb", "c_pbe_k333_e120", "gb, a = 10/ecutwfc"),
    "MgO_k333/pbe0": ("mgo_pbe0_k333_e150_gbx", "mgo_pbe_k333_e150", "default (= QE gbx)"),
    "MgO_k333/hse06": ("mgo_hse06_k333_e150_gb", "mgo_pbe_k333_e150", "gb, a = 10/ecutwfc"),
    "hBN_k661/pbe0": ("hbn_pbe0_k661_e100_gbx", "hbn_pbe_k661_e100", "default: padded slab"),
    "hBN_k661/pbe0_periodic": ("hbn_pbe0_k661_e100_gbx", "hbn_pbe_k661_e100", "periodic, gbx, a = 10/ecutwfc"),
    "hBN_k661/hse06": ("hbn_hse06_k661_e100_gb", "hbn_pbe_k661_e100", "default: padded slab"),
    "hBN_k661/hse06_periodic": ("hbn_hse06_k661_e100_gb", "hbn_pbe_k661_e100", "periodic, gb, a = 10/ecutwfc"),
    "O2_box/pbe0": ("o2_pbe0_k111_e150_gbx", "o2_pbe_k111_e150", "default (= QE gbx)"),
    "O2_box/hse06": ("o2_hse06_k111_e150_gb", "o2_pbe_k111_e150", "gb, a = 10/ecutwfc"),
}
PBE_CASES = {
    "Si_k222/pbe": "si_pbe_k222_e60", "Si_k444/pbe": "si_pbe_k444_e60", "Si_disp_k222/pbe": "si_disp_pbe_k222_e60",
    "C_k333/pbe": "c_pbe_k333_e120", "MgO_k333/pbe": "mgo_pbe_k333_e150", "hBN_k661/pbe": "hbn_pbe_k661_e100",
    "O2_box/pbe": "o2_pbe_k111_e150",
}
FORCE_CASES = ("Si_disp_k222/pbe", "Si_disp_k222/pbe0", "Si_disp_k222/hse06")


def _last(pattern: str, text: str, group: int = 1) -> float | None:
    found = re.findall(pattern, text, re.M)
    if not found:
        return None
    value = found[-1]
    return float(value if isinstance(value, str) else value[group - 1])


def _band_edges(text: str) -> tuple[float | None, float | None]:
    """VBM and CBM (eV) from the converged eigenvalues of every k-point (and spin channel) at the end of the log."""

    start = text.rfind("Converged eigenvalues by k-point")
    if start < 0:
        return None, None
    occupied, empty = [], []
    for line in text[start:].splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0].isdigit():  # state, Ry, eV, occupation (not the force tables after it)
            try:
                energy, occupation = float(fields[2]), float(fields[3])
            except ValueError:
                continue
            (occupied if occupation > 0.5 else empty).append(energy)
    if not occupied or not empty:
        return None, None
    return max(occupied), min(empty)


def parse(path: Path) -> dict:
    text = path.read_text()
    record: dict = {}
    hybrid = "Fock loop" in text
    if hybrid:
        record["energy"] = _last(r"Total Energy \([^)]*\)\s*=\s*(-?\d+\.\d+) \[Ry\]", text)
        record["fock"] = _last(r"Exact exchange a_x\*E_x\s*=\s*(-?\d+\.\d+) \[Ry\]\s*$", text)
        record["ex"] = _last(r"Exact exchange E_x\s*=\s*(-?\d+\.\d+)", text)
        record["outer"] = _last(r"Fock loop converged after (\d+) outer", text)
        solves = re.findall(
            r"Exact exchange: (\d+) pair-density Poisson solves in ([\d.]+) s \(([\d.]+) s in the solver\); "
            r"Fock loop ([\d.]+) s", text)
        if solves:
            pairs, seconds, solver, loop = solves[-1]
            record.update(pair_solves=int(pairs), exchange_s=float(seconds), solver_s=float(solver), fock_loop_s=float(loop))
    else:
        record["energy"] = _last(r"^\s*Total Energy\s*=\s*(-?\d+\.\d+) \[Ry\]", text)
    record["vbm"] = _last(r"Valence band maximum\s*=\s*-?\d+\.\d+ \[Ry\] =\s*(-?\d+\.\d+) \[eV\]", text)
    record["cbm"] = _last(r"Conduction band minimum\s*=\s*-?\d+\.\d+ \[Ry\] =\s*(-?\d+\.\d+) \[eV\]", text)
    record["gap"] = _last(r"Band gap \([a-z]+\)\s*=\s*-?\d+\.\d+ \[Ry\] =\s*(-?\d+\.\d+) \[eV\]", text)
    if record["gap"] is None:
        vbm, cbm = _band_edges(text)
        if vbm is not None:
            record.update(vbm=vbm, cbm=cbm, gap=cbm - vbm)
    record["wall_s"] = _last(r"Total (?:accelerated )?Python wall time \[sec\] :\s*([\d.]+)", text)
    record["scf_s"] = _last(r"SCF wall time\s*:\s*([\d.]+)", text)
    marker = text.rfind("coordinates and tot. forces (after setting net force to zero)")
    if marker >= 0:
        forces = []
        for line in text[marker:].splitlines()[4:]:
            fields = line.split()
            if len(fields) != 7 or not fields[0].isdigit():
                break
            forces.append(tuple(float(v) for v in fields[4:7]))
        record["forces"] = forces
    return record


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=None, help="a directory of run_examples.py --out")
    parser.add_argument("--runs", type=Path, default=None, help="runs.jsonl of the runner (peak device memory)")
    parser.add_argument("--json", type=Path, default=None, help="also write the parsed numbers to this file")
    arguments = parser.parse_args(argv)

    def log(case: str) -> Path:
        return (arguments.logs / case / "parsec.out") if arguments.logs else (HERE / case / "parsec_python.out")

    sys.path.insert(0, str(HERE))
    from run_examples import cases  # the inputs of the example in the order of the README

    results = {}
    for case in cases():
        if not case.startswith("_") and log(case).is_file():
            results[case] = parse(log(case))
    runs = {}
    if arguments.runs and arguments.runs.is_file():
        for line in arguments.runs.read_text().splitlines():
            record = json.loads(line)
            runs[record["case"]] = record
    if arguments.json:
        arguments.json.write_text(json.dumps(results, indent=1))

    def f(value, digits=8, sign=False):
        if value is None:
            return ""
        return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"

    def e(value):
        return "" if value is None else f"{value:+.1e}"

    print("## PBE\n")
    rows = []
    for case, row in PBE_CASES.items():
        if case not in results:
            continue
        r, q = results[case], QE[row]
        rows.append([case.split("/")[0], f(r["energy"]), f(q[0]), e(r["energy"] - q[0]), f(r["gap"], 4), f(q[2], 4)])
    print(table(["system", "E_PBE this code (Ry)", "E_PBE QE (Ry)", "difference (Ry)", "gap this code (eV)", "gap QE (eV)"], rows))

    print("\n## Hybrids against QE\n")
    rows = []
    for case, (row, pbe_row, setting) in CASES.items():
        system = case.split("/")[0]
        pbe_case = f"{system}/pbe"
        if case not in results or pbe_case not in results:
            continue
        r, p = results[case], results[pbe_case]
        q, qp = QE[row], QE[pbe_row]
        de, dq = r["energy"] - p["energy"], q[0] - qp[0]
        moved = ""
        if row in QE_MOVED and pbe_row in QE_MOVED:
            dm = QE_MOVED[row][0] - QE_MOVED[pbe_row][0]
            moved = e(de - dm)
        rows.append([system, case.split("/")[1], setting, f(de, 8, True), f(dq, 8, True), e(de - dq), moved,
                     f(r.get("fock")), f(q[1]), e((r.get("fock") or 0) - (q[1] or 0)), f(r["gap"], 4), f(q[2], 4)])
    print(table(["system", "input", "q+G=0 / slab", "dE this code", "dE QE", "ddE", "ddE (QE moved)", "Fock this code",
                 "Fock QE", "difference", "gap this code", "gap QE"], rows))

    print("\n## QE with the atoms moved off its grid\n")
    rows = []
    for case, (row, pbe_row, setting) in CASES.items():
        if row not in QE_MOVED or case.endswith("_default") or case.endswith("_periodic"):
            continue
        system = case.split("/")[0]
        dq = QE[row][0] - QE[pbe_row][0]
        dm = QE_MOVED[row][0] - QE_MOVED[pbe_row][0]
        mine = ""
        if case in results and f"{system}/pbe" in results:
            de = results[case]["energy"] - results[f"{system}/pbe"]["energy"]
            mine = [f(de, 8, True), e(de - dq), e(de - dm)]
        rows.append([system, case.split("/")[1], f(dq, 8, True), f(dm, 8, True), e(dm - dq), *(mine or ["", "", ""])])
    print(table(["system", "input", "dE QE", "dE QE moved", "change", "dE this code", "ddE", "ddE (moved)"], rows))

    print("\n## Forces (atom 2, x, Ry/bohr, net force removed)\n")
    rows = []
    pbe = results.get("Si_disp_k222/pbe")
    for case in FORCE_CASES:
        if case not in results or not results[case].get("forces"):
            continue
        row = PBE_CASES.get(case) or CASES[case][0]
        mine, theirs = results[case]["forces"][1][0], QE[row][3][0]
        extra = ""
        if pbe and case != "Si_disp_k222/pbe" and pbe.get("forces"):
            extra = e((mine - pbe["forces"][1][0]) - (theirs - QE["si_disp_pbe_k222_e60"][3][0]))
        rows.append([case.split("/")[1], f(mine, 6), f(theirs, 8), e(mine - theirs), extra])
    print(table(["functional", "this code", "QE", "difference", "hybrid - PBE: difference"], rows))

    print("\n## Timings\n")
    rows = []
    for case, r in results.items():
        run = runs.get(case, {})
        peak = run.get("peak_device_gib")
        rows.append([case, str(run.get("gpus", "")), f(r.get("wall_s"), 1), f(r.get("scf_s"), 1),
                     f(r.get("exchange_s"), 2), f(r.get("solver_s"), 2), str(r.get("pair_solves", "")),
                     "" if r.get("outer") is None else str(int(r["outer"])),
                     "" if not peak else f(max(peak.values()), 1)])
    print(table(["input", "GPUs", "program (s)", "SCF (s)", "exchange (s)", "solver (s)", "pair solves", "outer",
                 "peak GiB"], rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
