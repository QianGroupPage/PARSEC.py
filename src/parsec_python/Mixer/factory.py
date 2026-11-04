"""The mixer of ``Mixing_Method`` (``parsec.F90:1318-1369``)."""

from __future__ import annotations

import numpy as np

from ..models import MixingSettings
from .anderson import AndersonMixer
from .broyden import GeneralizedBroydenMixer
from .multisecant import MultisecantMixer


def make_mixer(
    settings: MixingSettings | None = None,
    weights: np.ndarray | None = None,
) -> AndersonMixer | GeneralizedBroydenMixer | MultisecantMixer:
    """Return a fresh mixer for ``settings.method``, as PARSEC selects it.

    ``anderson`` gives :class:`AndersonMixer` unchanged (``anderson.f90``),
    ``broyden`` :class:`GeneralizedBroydenMixer` (``genbro.f90``) and
    ``msecant1``/``msecant2``/``msecant3`` :class:`MultisecantMixer`
    (``msecant1-3.f90``).  Every mixer has ``mix(x_in, x_out, iteration=k)``
    and ``reset()``.  ``weights`` are the inner-product weights of the Broyden
    and multisecant mixers (the orbit multiplicities of a symmetry wedge);
    the Anderson mixer of the wedge is
    :class:`parsec_python.acceleration.SCF.SymmetryAndersonMixer`, so this
    function refuses weights for it.
    """

    settings = MixingSettings() if settings is None else settings
    method = settings.method
    if method == "anderson":
        if weights is not None:
            raise ValueError(
                "AndersonMixer works on full vectors; the weighted Anderson mixer "
                "of a symmetry wedge is SymmetryAndersonMixer"
            )
        return AndersonMixer(settings)
    if method == "broyden":
        return GeneralizedBroydenMixer(settings, weights=weights)
    if method in {"msecant1", "msecant2", "msecant3"}:
        return MultisecantMixer(settings, weights=weights)
    raise ValueError(f"unknown mixing method {method!r}")


__all__ = ["make_mixer"]
