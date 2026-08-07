"""
Mixer dispatcher module

This module provides the mixing interface for SCF iterations.
Default mixer is msecant1 (Type-I multi-secant method).

Usage:
    from old_architecture.Mixer.mixer import mixer, reset_mixer

    # Before SCF loop:
    reset_mixer()

    # Inside SCF loop:
    pot = mixer(pot, potNew - pot)
"""

from . import msecant1

# Try to import optional mixers, but don't fail if they don't exist
try:
    from . import msecant2
except ImportError:
    msecant2 = None

try:
    from . import msecant3
except ImportError:
    msecant3 = None

try:
    from . import simplemix
except ImportError:
    simplemix = None


# Global mixer instances
_mixers = {}

# Default mixer name
_default_mixer = "msecant1"


def _get_instance(name):
    """Get or create mixer instance by name."""
    if name not in _mixers:
        if name == "msecant1":
            _mixers[name] = msecant1.mixer()
        elif name == "msecant2" and msecant2 is not None:
            _mixers[name] = msecant2.mixer()
        elif name == "msecant3" and msecant3 is not None:
            _mixers[name] = msecant3.mixer()
        elif name == "simplemix":
            _mixers[name] = None  # simplemix handled separately
        else:
            raise ValueError(f"Unknown mixer '{name}'")
    return _mixers[name]


def reset_mixer(name=None):
    """
    Reset mixer state.
    
    Call this once before the SCF loop to clear persistent variables.
    If name is None, resets the default mixer.
    """
    global _mixers
    if name is None:
        name = _default_mixer
    
    # Remove existing instance to force re-creation
    if name in _mixers:
        del _mixers[name]
    
    # Also reset via the module-level function if it exists
    if name == "msecant1":
        msecant1._persistent_mixer = None


def mixer(x1, f1, name=None):
    """
    Mixer function
    
    [x_new, m] = mixer(x1, f1)
    
    Parameters:
        x1: Current iterate (potential vector)
        f1: Residual f(x1) = potNew - pot
        name: Mixer name (default: 'msecant1')
    
    Returns:
        x_new: New estimate of the solution
        m: Number of secant equations used
    
    To change the mixer, modify _default_mixer or pass name argument.
    Available mixers: 'msecant1', 'msecant2', 'msecant3', 'simplemix'
    """
    if name is None:
        name = _default_mixer
    
    if name == "simplemix" and simplemix is not None:
        return simplemix.simplemix(x1, f1), 0
    
    inst = _get_instance(name)
    if inst is None:
        raise ValueError(f"Mixer '{name}' not available")
    
    return inst.mixer(x1, f1)
