"""``bak_complete_parsec.in``: the effective input of a run, written beside the input (PARSEC 1.5).

The text is made by the input reader (:mod:`parsec_python.Input.complete_input`, ``Generate_Complete_Parsec_In``);
this module writes it, as PARSEC 1.5's ``write_out_parameters`` (``usrinputfile.F90:5477-6042``) writes it in the run
directory.  Both command lines call :func:`write_complete_input` before the calculation.
"""

from __future__ import annotations

from pathlib import Path

from ..Input.complete_input import COMPLETE_INPUT_NAME


def write_complete_input(translation) -> list[str]:
    """Write ``bak_complete_parsec.in`` beside the input; the lines for parsec.out.

    Nothing is written without ``Generate_Complete_Parsec_In`` (``translation.complete_input`` is ``None``), or
    where the input itself is named ``bak_complete_parsec.in``.
    """

    text = getattr(translation, "complete_input", None)
    if text is None:
        return []
    source = Path(translation.source)
    path = source.parent / COMPLETE_INPUT_NAME
    if path.resolve() == source.resolve():
        return [f" WARNING: {COMPLETE_INPUT_NAME} is not written: it is the input of this run"]
    path.write_text(text, encoding="utf-8", newline="\n")
    lines = [f" Writing out all processed input parameters to {path}"]
    # the warning of the header (a path with a comment character of ESDF), once in parsec.out
    notes = [line[len("# WARNING: ") :] for line in text.splitlines() if line.startswith("# WARNING: ")]
    if notes:
        lines.append(" WARNING: " + " ".join(notes))
    return lines


__all__ = ["write_complete_input"]
