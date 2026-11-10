"""Run the molecules of this example with libxc's functionals and compare them with the code's own and with QE.

    PARSEC_LIBXC_LIBRARY=.../libxc.so python examples/0d_hybrid_functionals/libxc/run_libxc.py \
        --out OUT [--molecules H2 H2O CH4 CO N2] [--cases pbe libxc_pbe ...] [--gpus 1]

Every case copies ``<molecule>/<source>/`` of this example to ``OUT/<molecule>_<case>/``, replaces its
``Correlation_Type`` line by the libxc keyword of the case (the code's own cases keep the input) and runs
``src/parsec_python/main.py`` there with ``--pp-dir`` of this example.  ``OUT/summary.json`` holds the total
energy and ``a_x E_x`` of each run.  The cases:

==============  ========  ===========================================
case            source    functional
==============  ========  ===========================================
``pbe``         pbe       the code's PBE
``libxc_pbe``   pbe       ``gga_x_pbe gga_c_pbe``
``b3lyp``       b3lyp     the code's B3LYP
``libxc_b3lyp`` b3lyp     ``hyb_gga_xc_b3lyp`` (402)
``pbe0``        pbe0      the code's PBE0
``libxc_pbeh``  pbe0      ``hyb_gga_xc_pbeh`` (406)
``hse06``       hse06     the code's HSE06 (QE's form)
``libxc_hse06`` hse06     ``hyb_gga_xc_hse06`` (428; QE's ``hse06_libxc`` row)
``libxc_hse03`` hse03     ``hyb_gga_xc_hse03`` (427; QE's ``hse03_libxc`` row)
==============  ========  ===========================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

EXAMPLE = Path(__file__).resolve().parents[1]
REPOSITORY = EXAMPLE.parents[1]
CASES = {
    "pbe": ("pbe", None),
    "libxc_pbe": ("pbe", "LibXC_Functional: gga_x_pbe gga_c_pbe"),
    "b3lyp": ("b3lyp", None),
    "libxc_b3lyp": ("b3lyp", "LibXC_Functional: hyb_gga_xc_b3lyp"),
    "pbe0": ("pbe0", None),
    "libxc_pbeh": ("pbe0", "LibXC_Functional: hyb_gga_xc_pbeh"),
    "hse06": ("hse06", None),
    "libxc_hse06": ("hse06", "LibXC_Functional: hyb_gga_xc_hse06"),
    "libxc_hse03": ("hse03", "LibXC_Functional: hyb_gga_xc_hse03"),
}
_TOTAL = re.compile(r"^\s*Total Energy(?: \([^)]*\))?\s*=\s*(-?\d+\.\d+)\s*\[Ry\]", re.MULTILINE)
_FOCK = re.compile(r"^\s*Exact exchange a_x\*E_x\s*=\s*(-?\d+\.\d+)", re.MULTILINE)


def prepare(molecule: str, case: str, out: Path) -> Path:
    source, line = CASES[case]
    target = out / f"{molecule}_{case}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(EXAMPLE / molecule / source, target, ignore=shutil.ignore_patterns("*.out", "*.dat"))
    path = target / "parsec.in"
    text = path.read_text()
    if line is not None:
        text, count = re.subn(r"^Correlation_Type:.*$", line, text, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"{path}: no Correlation_Type line")
    path.write_text(text)
    return path


def run(path: Path, gpus: int) -> dict:
    started = time.perf_counter()
    command = [
        sys.executable,
        str(REPOSITORY / "src" / "parsec_python" / "main.py"),
        str(path),
        "--pp-dir",
        str(EXAMPLE),
        "--no-archive",
        "--gpus",
        str(gpus),
    ]
    completed = subprocess.run(command, cwd=path.parent, capture_output=True, text=True)
    (path.parent / "stdout.txt").write_text(completed.stdout[-20000:] + "\n" + completed.stderr[-20000:])
    log = path.parent / "parsec.out"
    text = log.read_text() if log.exists() else ""
    totals = _TOTAL.findall(text)
    focks = _FOCK.findall(text)
    return {
        "exit": completed.returncode,
        "seconds": time.perf_counter() - started,
        "total": float(totals[-1]) if totals else None,
        "fock": float(focks[-1]) if focks else None,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--molecules", nargs="+", default=["H2", "H2O", "CH4", "CO", "N2"])
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--gpus", type=int, default=1)
    arguments = parser.parse_args(argv)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for molecule in arguments.molecules:
        for case in arguments.cases:
            record = run(prepare(molecule, case, out), arguments.gpus)
            summary[f"{molecule}/{case}"] = record
            print(f"{molecule:5s} {case:12s} exit {record['exit']}  {record['seconds']:7.1f} s  "
                  f"E = {record['total']}  a_x E_x = {record['fock']}", flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
