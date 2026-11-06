"""Archives (``.npz``) that a test reads: every array read out, the file closed.

``np.load`` of an ``.npz`` returns an ``NpzFile`` that holds the file open until it is closed or
collected.  A test that keeps one in a variable while its temporary folder is removed (the folder of a
``with tempfile.TemporaryDirectory()`` block ends before the frame that holds the variable) fails on
Windows with ``PermissionError: [WinError 32]``: an open file cannot be deleted there.
"""

from __future__ import annotations

import numpy as np


class ReadArchive(dict):
    """The arrays of an archive by name; ``files`` lists the names, as ``NpzFile.files`` does."""

    @property
    def files(self) -> list[str]:
        return list(self)


def read_archive(path, **options) -> ReadArchive:
    """Every array of the ``.npz`` file ``path`` (``options`` go to ``np.load``), the file closed."""

    with np.load(path, **options) as archive:
        return ReadArchive((name, archive[name]) for name in archive.files)


__all__ = ["ReadArchive", "read_archive"]
