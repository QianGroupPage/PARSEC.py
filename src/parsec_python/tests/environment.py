"""Environment variables a test sets for itself and puts back when it ends.

A variable written straight into ``os.environ`` stays for every later module of the suite: up to
2026-11-05 four modules (``test_misc_parity``, ``test_parity_combinations``, ``test_parity_sweepa``,
``test_sweepb_parity``) called ``os.environ.setdefault("PARSEC_KPOINT_BACKEND", "cpu")``, and every
module after the first of them ran its k-point solves and periodic Hartree solves on the host in a
full suite, on a GPU when it ran alone.
"""

from __future__ import annotations

import os
from unittest import mock

KPOINT_BACKEND = "PARSEC_KPOINT_BACKEND"


def kpoints_on_the_cpu():
    """``PARSEC_KPOINT_BACKEND=cpu`` inside the block (a value the caller set wins, as with
    ``setdefault``), and the former state after it: the k-point eigensolver, and the periodic Hartree
    solve that follows its plan, stay on the host."""

    return mock.patch.dict(os.environ, {KPOINT_BACKEND: os.environ.get(KPOINT_BACKEND, "cpu")})


def pin_kpoints_on_the_cpu(case) -> None:
    """:func:`kpoints_on_the_cpu` for the rest of the test ``case`` (its cleanups put it back)."""

    patcher = kpoints_on_the_cpu()
    patcher.start()
    case.addCleanup(patcher.stop)


__all__ = ["KPOINT_BACKEND", "kpoints_on_the_cpu", "pin_kpoints_on_the_cpu"]
