"""The tables of the README from the logs of this example and the Quantum ESPRESSO numbers below.

    python examples/0d_meta_gga/collect_results.py                    # the logs parsec_python.out of the example
    python examples/0d_meta_gga/collect_results.py --runs RUNS        # RUNS/<case>/parsec.out and RUNS/runs.jsonl

Prints Markdown: the PBE level, the meta-GGAs (E - E_PBE in mRy, each code against its own PBE; HOMO), the grid
spacing 0.13 bohr against 0.16, the meta-GGA hybrids (E - E_PBE, a_x E_x, HOMO), the forces of the distorted water
molecule and the costs (iterations, wall time; peak device memory with RUNS/runs.jsonl).

QE: Quantum ESPRESSO 7.5 + libxc 7.0.0, the same FHI98PP potentials (UPF), 20 bohr box, Martyna-Tuckerman, dual 4;
the inputs are in ``qe/`` and the numbers in ``QE`` below, for each row at the cutoff of ``qe_cutoff`` (300 Ry
where QE's convergence rows exist, TPSS of H2O and O2 250 Ry, else 150 Ry); E - E_PBE takes QE's PBE at the same
cutoff.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOLECULES = ["H2O", "CH4", "NH3", "N2", "CO", "O2"]
SEMILOCAL = ["scan", "r2scan", "tpss"]
HYBRIDS = ["tpssh", "r2scan0"]
NAMES = {"pbe": "PBE", "scan": "SCAN", "r2scan": "r2SCAN", "tpss": "TPSS", "tpssh": "TPSSh", "r2scan0": "r2SCAN0"}

# QE: E (Ry), HOMO (eV; O2: up, down), Fock = a_x E_x (Ry), forces (Ry/bohr, atoms O, H1, H2), at ecutwfc (Ry)
QE = {
    "H2O/pbe/300": {"E": -34.34607194, "homo": -7.2383},
    "H2O/r2scan/300": {"E": -34.34809558, "homo": -7.5688},
    "H2O/scan/300": {"E": -34.34431024, "homo": -7.5973},
    "H2O/pbe/250": {"E": -34.34607214, "homo": -7.2384},
    "H2O/tpss/250": {"E": -34.35715479, "homo": -7.3158},
    "H2O/pbe/150": {"E": -34.34601171, "homo": -7.2381},
    "H2O/tpssh/150": {"E": -34.35419783, "homo": -8.0323, "fock": -0.78721403},
    "H2O/r2scan0/150": {"E": -34.33470918, "homo": -9.2906, "fock": -1.97347758},
    "CH4/pbe/300": {"E": -16.15718789, "homo": -9.4644},
    "CH4/r2scan/300": {"E": -16.1586312, "homo": -9.8132},
    "CH4/scan/300": {"E": -16.1611123, "homo": -9.8252},
    "CH4/pbe/150": {"E": -16.15717012, "homo": -9.4645},
    "CH4/tpss/150": {"E": -16.19652752, "homo": -9.6566},
    "CH4/tpssh/150": {"E": -16.19859597, "homo": -10.2556, "fock": -0.59342921},
    "CH4/r2scan0/150": {"E": -16.16091509, "homo": -11.2306, "fock": -1.48112872},
    "NH3/pbe/300": {"E": -23.46695167, "homo": -6.1604},
    "NH3/r2scan/300": {"E": -23.46212937, "homo": -6.4625},
    "NH3/scan/300": {"E": -23.46043556, "homo": -6.4947},
    "NH3/pbe/150": {"E": -23.46692688, "homo": -6.1603},
    "NH3/tpss/150": {"E": -23.48456702, "homo": -6.2282},
    "NH3/tpssh/150": {"E": -23.48380036, "homo": -6.8434, "fock": -0.67180198},
    "NH3/r2scan0/150": {"E": -23.45576923, "homo": -7.9234, "fock": -1.68127045},
    "O2/pbe/300": {"E": -63.71092564, "homo": [-6.8741, -11.6063]},
    "O2/r2scan/300": {"E": -63.69745485, "homo": [-7.4424, -11.6907]},
    "O2/scan/300": {"E": -63.69267764, "homo": [-7.5098, -11.6164]},
    "O2/pbe/250": {"E": -63.71092361, "homo": [-6.874, -11.6063]},
    "O2/tpss/250": {"E": -63.7091373, "homo": [-7.0621, -11.5653]},
    "N2/pbe/150": {"E": -39.83021459, "homo": -10.2959},
    "N2/scan/150": {"E": -39.79094194, "homo": -10.8024},
    "N2/r2scan/150": {"E": -39.79497485, "homo": -10.7267},
    "N2/tpss/150": {"E": -39.82182266, "homo": -10.4621},
    "N2/tpssh/150": {"E": -39.81179559, "homo": -11.207, "fock": -0.89752649},
    "N2/r2scan0/150": {"E": -39.76829315, "homo": -12.4856, "fock": -2.24536486},
    "CO/pbe/150": {"E": -43.31651727, "homo": -9.0616},
    "CO/scan/150": {"E": -43.29369556, "homo": -9.5976},
    "CO/r2scan/150": {"E": -43.29678114, "homo": -9.5134},
    "CO/tpss/150": {"E": -43.31492859, "homo": -9.2832},
    "CO/tpssh/150": {"E": -43.3067223, "homo": -9.9371, "fock": -0.9359389},
    "CO/r2scan0/150": {"E": -43.27246139, "homo": -11.0414, "fock": -2.34421579},
    "H2O_dist/pbe/150": {"E": -34.32683509, "forces": [[0.15095013, 0.0, 0.05496249], [-0.1234476, 0.0, -0.08658127],
                                                       [-0.02750253, 0.0, 0.03161879]]},
    "H2O_dist/pbe/200": {"E": -34.32688382, "forces": [[0.15100303, 0.0, 0.05501079], [-0.12348078, 0.0, -0.08660788],
                                                       [-0.02752225, 0.0, 0.0315971]]},
    "H2O_dist/r2scan/150": {"E": -34.32432523, "forces": [[0.14885579, 0.0, 0.07302874], [-0.13673486, 0.0, -0.09500094],
                                                          [-0.01212093, 0.0, 0.02197219]]},
    "H2O_dist/r2scan/200": {"E": -34.32435058, "forces": [[0.14905271, 0.0, 0.0732812], [-0.13685422, 0.0, -0.09512237],
                                                          [-0.01219848, 0.0, 0.02184117]]},
}


def qe_cutoff(molecule: str, functional: str) -> int:
    """The cutoff (Ry) of the QE row compared with: the converged rows where the reference has them."""

    if functional in HYBRIDS or molecule in ("N2", "CO"):
        return 150
    if functional == "tpss":
        return 250 if molecule in ("H2O", "O2") else 150
    return 300


_TOTAL = re.compile(r"^\s*Total Energy(?: \([^)]*\))?\s*=\s*(-?\d+\.\d+)\s*\[Ry\]", re.MULTILINE)
_FOCK = re.compile(r"^\s*Exact exchange a_x\*E_x\s*=\s*(-?\d+\.\d+)\s*\[Ry\]\s*$", re.MULTILINE)
_SCF = re.compile(r"^\s*SCF iter #\s*(\d+)", re.MULTILINE)
_OUTER = re.compile(r"Fock loop iteration\s+\d+ \([^)]*\): inner SCF of (\d+) iterations")
_START = re.compile(r"(\d+) SCF iterations of the \S+ start")
_WALL = re.compile(r"Total (?:accelerated )?Python wall time \[sec\]\s*:\s*([0-9.]+)")
_TABLE = re.compile(r"^\s*(up |dn )?State\s+Eigenvalue \[Ry\]\s+Eigenvalue \[eV\]\s+Occup\.", re.MULTILINE)
_FORCES = "coordinates and tot. forces (after setting net force to zero)"


def eigenvalue_tables(text: str) -> dict[str, list[tuple[float, float]]]:
    """The last eigenvalue table of each channel ("" or "up", "dn"): (eV, occupation) per state."""

    tables: dict[str, list[tuple[float, float]]] = {}
    for match in _TABLE.finditer(text):
        rows = []
        for line in text[match.end():].split("\n")[1:]:
            fields = line.split()
            if not fields:
                if rows:
                    break
                continue
            if not fields[0].isdigit():
                break
            rows.append((float(fields[2]), float(fields[3])))
        tables[(match.group(1) or "").strip()] = rows
    return tables


def homo(text: str):
    """HOMO (eV): one value, or (up, down) for a spin-polarized run."""

    tables = eigenvalue_tables(text)
    values = {channel: max(e for e, occupation in rows if occupation > 0.5) for channel, rows in tables.items() if rows}
    if "up" in values:
        return (values["up"], values["dn"])
    return values.get("")


def forces(text: str) -> list[list[float]] | None:
    """Total forces after the net force is removed (Ry/bohr), in the atom order of the input."""

    start = text.rfind(_FORCES)
    if start < 0:
        return None
    rows = []
    for line in text[start:].split("\n")[1:]:
        fields = line.split()
        if len(fields) == 7 and fields[0].isdigit():
            rows.append([float(value) for value in fields[4:]])
        elif rows:
            break
    return rows


def gradient_form(text: str) -> str | None:
    match = re.search(r"(orbital|density) form", text)
    return match.group(1) if match else None


def read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    totals = _TOTAL.findall(text)
    focks = _FOCK.findall(text)
    walls = _WALL.findall(text)
    start = _START.search(text)
    return {
        "E": float(totals[-1]) if totals else None,
        "fock": float(focks[-1]) if focks else None,
        "homo": homo(text),
        "forces": forces(text),
        "converged": "Self-consistency convergence achieved" in text,
        "scf": int(_SCF.findall(text)[-1]) if _SCF.findall(text) else None,
        "inner": [int(value) for value in _OUTER.findall(text)],
        "start": int(start.group(1)) if start else None,
        "form": gradient_form(text),
        "wall": float(walls[-1]) if walls else None,
    }


def load(runs: Path | None) -> tuple[dict, dict]:
    logs, costs = {}, {}
    systems, variants = MOLECULES + ["H2O_dist"], ["pbe"] + SEMILOCAL + HYBRIDS + ["pbe_h013", "r2scan_h013"]

    def rank(path: Path) -> tuple:
        system, variant = path.parent.relative_to(HERE).parts
        return (systems.index(system) if system in systems else len(systems), system,
                variants.index(variant) if variant in variants else len(variants), variant)

    for path in sorted(HERE.glob("*/*/parsec.in"), key=rank):
        case = path.parent.relative_to(HERE).as_posix()
        log = (runs / case / "parsec.out") if runs is not None else (path.parent / "parsec_python.out")
        record = read(log)
        if record is not None:
            logs[case] = record
    if runs is not None and (runs / "runs.jsonl").is_file():
        for line in (runs / "runs.jsonl").read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                costs[record["case"]] = record
    return logs, costs


def _homo_text(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (tuple, list)):
        return f"{value[0]:.4f} / {value[1]:.4f}"
    return f"{value:.4f}"


def _homo_difference(ours, theirs) -> str:
    """meV; O2: the up channel (the HOMO), the down channel's highest occupied level after the slash."""

    if ours is None or theirs is None:
        return "-"
    if isinstance(ours, (tuple, list)):
        return f"{1000 * (ours[0] - theirs[0]):+.1f} / {1000 * (ours[1] - theirs[1]):+.1f}"
    return f"{1000 * (ours - theirs):+.1f}"


def _difference(logs: dict, case: str, reference: str) -> float | None:
    if case in logs and reference in logs and logs[case]["E"] is not None and logs[reference]["E"] is not None:
        return logs[case]["E"] - logs[reference]["E"]
    return None


def table_pbe(logs: dict) -> list[str]:
    lines = ["| molecule | E_PBE this code (Ry) | E_PBE QE (Ry) | QE cutoff (Ry) | difference (Ry) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) |",
             "|---|---|---|---|---|---|---|---|"]
    for molecule in MOLECULES:
        ours = logs.get(f"{molecule}/pbe")
        cutoff = 150 if molecule in ("N2", "CO") else 300
        theirs = QE[f"{molecule}/pbe/{cutoff}"]
        if ours is None:
            continue
        lines.append(f"| {molecule} | {ours['E']:.8f} | {theirs['E']:.8f} | {cutoff} | {ours['E'] - theirs['E']:+.2e} | "
                     f"{_homo_text(ours['homo'])} | {_homo_text(theirs['homo'])} | {_homo_difference(ours['homo'], theirs['homo'])} |")
    return lines


def table_semilocal(logs: dict) -> list[str]:
    lines = ["| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | QE cutoff (Ry) | this code - QE (mRy) | "
             "HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | SCF iterations | gradient form |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for molecule in MOLECULES:
        for functional in SEMILOCAL:
            case = f"{molecule}/{functional}"
            if case not in logs:
                continue
            ours = logs[case]
            difference = _difference(logs, case, f"{molecule}/pbe")
            cutoff = qe_cutoff(molecule, functional)
            theirs = QE[f"{molecule}/{functional}/{cutoff}"]
            reference = theirs["E"] - QE[f"{molecule}/pbe/{cutoff}"]["E"]
            note = "" if ours["converged"] else " (Max_Iter)"
            lines.append(
                f"| {molecule} | {NAMES[functional]}{note} | {1000 * difference:+.3f} | {1000 * reference:+.3f} | {cutoff} | "
                f"{1000 * (difference - reference):+.3f} | {_homo_text(ours['homo'])} | {_homo_text(theirs['homo'])} | "
                f"{_homo_difference(ours['homo'], theirs['homo'])} | {ours['scf']} | {ours['form'] or '-'} |")
    return lines


def table_spacing(logs: dict) -> list[str]:
    lines = ["| molecule | h (bohr) | E_r2SCAN - E_PBE this code (mRy) | QE 300 Ry (mRy) | this code - QE (mRy) | HOMO this code (eV) | dHOMO (meV) | SCF iterations |",
             "|---|---|---|---|---|---|---|---|"]
    for molecule in ("H2O", "O2"):
        theirs = QE[f"{molecule}/r2scan/300"]
        reference = theirs["E"] - QE[f"{molecule}/pbe/300"]["E"]
        for suffix, spacing in (("", "0.16"), ("_h013", "0.13")):
            case = f"{molecule}/r2scan{suffix}"
            difference = _difference(logs, case, f"{molecule}/pbe{suffix}")
            if difference is None:
                continue
            ours = logs[case]
            lines.append(f"| {molecule} | {spacing} | {1000 * difference:+.3f} | {1000 * reference:+.3f} | "
                         f"{1000 * (difference - reference):+.3f} | {_homo_text(ours['homo'])} | "
                         f"{_homo_difference(ours['homo'], theirs['homo'])} | {ours['scf']} |")
    return lines


def table_hybrids(logs: dict) -> list[str]:
    lines = ["| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | this code - QE (mRy) | a_x E_x this code (Ry) | "
             "a_x E_x QE (Ry) | difference (mRy) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | inner SCF per outer iteration |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for molecule in MOLECULES:
        for functional in HYBRIDS:
            case = f"{molecule}/{functional}"
            if case not in logs:
                continue
            ours = logs[case]
            difference = _difference(logs, case, f"{molecule}/pbe")
            theirs = QE[f"{molecule}/{functional}/150"]
            reference = theirs["E"] - QE[f"{molecule}/pbe/150"]["E"]
            lines.append(
                f"| {molecule} | {NAMES[functional]} | {1000 * difference:+.3f} | {1000 * reference:+.3f} | "
                f"{1000 * (difference - reference):+.3f} | {ours['fock']:.8f} | {theirs['fock']:.8f} | "
                f"{1000 * (ours['fock'] - theirs['fock']):+.3f} | {_homo_text(ours['homo'])} | {_homo_text(theirs['homo'])} | "
                f"{_homo_difference(ours['homo'], theirs['homo'])} | {ours['inner']} |")
    return lines


def table_forces(logs: dict) -> list[str]:
    lines = ["| functional | atom | this code (x, z) | QE 200 Ry (x, z) | this code - QE 200 Ry (x, z) | QE 150 - QE 200 Ry (x, z) |",
             "|---|---|---|---|---|---|"]
    order = [(2, 0, "O"), (0, 1, "H1 (1.10 A)"), (1, 2, "H2")]  # (row of this code, row of QE, name)
    for functional in ("pbe", "r2scan"):
        ours = logs.get(f"H2O_dist/{functional}")
        if ours is None or ours["forces"] is None:
            continue
        top, low = QE[f"H2O_dist/{functional}/200"]["forces"], QE[f"H2O_dist/{functional}/150"]["forces"]
        for mine, theirs, name in order:
            f, q, p = ours["forces"][mine], top[theirs], low[theirs]
            lines.append(f"| {NAMES[functional]} | {name} | {f[0]:+.6f}, {f[2]:+.6f} | {q[0]:+.6f}, {q[2]:+.6f} | "
                         f"{f[0] - q[0]:+.1e}, {f[2] - q[2]:+.1e} | {p[0] - q[0]:+.1e}, {p[2] - q[2]:+.1e} |")
    return lines


def table_costs(logs: dict, costs: dict) -> list[str]:
    """Wall time of the program (s) / SCF iterations (hybrids: PBE start + inner iterations of the Fock loop) /
    peak device memory (GiB, with RUNS/runs.jsonl) of each input."""

    variants = ["pbe", "scan", "r2scan", "tpss", "tpssh", "r2scan0"]
    lines = ["| input | " + " | ".join(NAMES[variant] for variant in variants) + " |",
             "|---|" + "---|" * len(variants)]
    rows = [(molecule, "", "0.16") for molecule in MOLECULES] + [
        ("H2O", "_h013", "0.13"), ("O2", "_h013", "0.13"), ("H2O_dist", "", "0.16")]
    for molecule, suffix, spacing in rows:
        cells = []
        for variant in variants:
            case = f"{molecule}/{variant}{suffix}"
            record = logs.get(case)
            if record is None:
                cells.append("")
                continue
            iterations = (f"{record['start']} + {sum(record['inner'])}" if record["inner"]
                          else f"{record['scf']}" + ("" if record["converged"] else " (Max_Iter)"))
            memory = costs.get(case, {}).get("peak_device_mib")
            wall = f"{record['wall']:.0f} s" if record["wall"] is not None else "-"
            cells.append(f"{wall} / {iterations}" + (f" / {memory / 1024:.1f} GiB" if memory else ""))
        name = molecule if molecule != "H2O_dist" else "H2O distorted (forces)"
        lines.append(f"| {name}, h {spacing} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=None, help="a run folder of run_examples.py")
    arguments = parser.parse_args(argv)
    logs, costs = load(arguments.runs)
    for title, lines in (("PBE level", table_pbe(logs)), ("Meta-GGAs", table_semilocal(logs)),
                         ("Grid spacing (r2SCAN)", table_spacing(logs)), ("Meta-GGA hybrids", table_hybrids(logs)),
                         ("Forces of the distorted water molecule (Ry/bohr, net force removed)", table_forces(logs)),
                         ("Costs (one A100)", table_costs(logs, costs))):
        print(f"### {title}\n")
        print("\n".join(lines))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
