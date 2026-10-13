"""One time step of PARSEC's molecular dynamics.

Translation of subroutine ``moldyn`` (``moldyn.f90``, modified by M. Alemany,
2003) for an isolated cluster.  One call of :func:`moldyn_step` is one call of
the Fortran routine: it takes the forces of the current structure and returns
the next structure.

Equations of motion
-------------------
``moldyn`` works in Hartree atomic units: lengths in bohr, masses in electron
masses, time in units of 2.4188843e-17 s.  The forces of PARSEC are in
Ry/bohr, so the acceleration of atom ``j`` is ``a_j = F_j/(2 m_j)``.

Friction coefficient ``beta < 1e-6`` (microcanonical ensemble): Newton's
equations, integrated with Beeman's variant of the Verlet algorithm
[J. Comput. Phys. 20, 130 (1976)], in which ``r`` and the accelerations of
the present and of the previous step give

``v(t)    = [r(t) - r(t-dt)]/dt + dt [2 a(t) + a(t-dt)]/6``
``r(t+dt) = r(t) + dt v(t) + dt**2 [4 a(t) - a(t-dt)]/6``

and the velocity kept for the restart file is the estimate
``v(t) + dt [3 a(t) - a(t-dt)]/2``.  The velocity of the centre of mass of
the movable atoms is subtracted from ``v(t)`` in every step.

Otherwise (canonical ensemble): the Langevin equation
``dv/dt = F/m - beta v + G/m`` with a Gaussian random acceleration of
standard deviation ``sqrt(2 beta kT/(m dt))`` per Cartesian component, where
``T`` is the target temperature of the step [Biswas and Hamann, Phys. Rev. B
34, 895 (1986)].  The same Beeman formulas are used with
``a = F/(2m) - beta v_old + random`` and the previous velocity estimate in
the friction term.  The centre of mass is not held in this branch.

The first call of a run that is not a restart has no previous step.  It draws
Maxwell velocities at the initial temperature, removes the centre-of-mass
velocity, rescales to the exact temperature and takes one step
``r(dt) = r(0) + dt v(0) + dt**2 a(0)/2``.

Atoms that must not move (``clust%mvat = 0``) have their velocity and
acceleration multiplied by zero.  They still consume random numbers.

Arithmetic
----------
Every expression is evaluated in the order of the Fortran source and the
sums run over the atoms in the order of the Fortran loops, so that a
trajectory agrees with the Fortran one to the last bit as long as the
forces and the random numbers do.

Behaviour of the Fortran routine that is reproduced on purpose
--------------------------------------------------------------
* The cluster is not recentred.  ``moldyn`` computes the shift that would
  centre the atoms and then overwrites the shifted coordinates with the
  unshifted ones, so a Langevin run can drift away from the grid centre.
* The kinetic energy that is reported is per *movable* atom, the potential
  energy per atom; their sum is the "total energy" of ``md_nrg.dat``.
* The reported temperature counts ``3 N`` degrees of freedom in the
  microcanonical branch although the centre of mass is held, and
  ``3 (N - 1)`` in the Langevin branch although it is not.  With a single
  movable atom the Langevin temperature is infinite.
* The first velocity rescaling tests ``beta < 1e-5`` while the integrator
  tests ``beta < 1e-6``.
* In the microcanonical branch the centre of mass of the *movable* atoms is
  held even when other atoms are fixed, although the fixed atoms exert a net
  force on it; the energy error is then of first order in the time step.
* ``Step_num`` time steps need ``Step_num + 1`` calls, hence ``Step_num + 1``
  self-consistent calculations; the structure returned by the last call is
  only written to the files.

Differences from the Fortran routine
------------------------------------
* Nothing is written: the lines of the four output files are produced by
  the functions of ``md_files`` from the returned values.
* The state is a value.  A step returns a new state and leaves its input
  untouched; the random source is the only object that is advanced.
* Only an isolated cluster is treated.  ``moldyn`` ends with
  ``back_to_cell``, which wraps the atoms of a periodic system into the cell
  and returns a cluster unchanged; it is left out.
* In a fresh run the velocities and accelerations of the state are zero
  before the first step, where the Fortran arrays are allocated but not yet
  assigned.  The first step does not read them.
* A state without any movable atom is refused; the Fortran routine divides
  by the mass of the movable atoms and returns not-a-number.
* Random numbers: see ``random_numbers``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from .random_numbers import UniformSource, gaussian_deviate
from .settings import (
    MICROCANONICAL_FRICTION,
    RESCALE_FRICTION,
    TIME_AU_TO_PS,
    MolecularDynamicsSettings,
)

#: ``rydberg`` of ``const.f90``: PARSEC's conversion from Ry to eV.
RYDBERG_EV = 13.6058
#: ``tempkry`` of ``const.f90``: one kelvin in Ry.
KELVIN_RYDBERG = 6.33327186e-06
#: ``angs`` of ``const.f90``: one bohr in angstrom.
BOHR_ANGSTROM = 0.529177


def _readonly(values: np.ndarray | Sequence, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, not {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class MolecularDynamicsState:
    """What PARSEC keeps between two calls of ``moldyn``.

    The arrays are those of ``type molecular_dynamic`` with the three
    Cartesian components side by side; all are read-only.

    ``masses``
        ``clust%amass``, shape ``(atoms,)``, electron masses.
    ``movable``
        ``clust%mvat``, shape ``(atoms,)``, 1 for an atom that moves, 0 for a
        fixed one.
    ``current_positions``
        ``xcur, ycur, zcur``, shape ``(atoms, 3)``, bohr: the structure whose
        forces the next call expects.
    ``previous_positions``
        ``xold, yold, zold``: the structure one step earlier.
    ``velocities``
        ``vxold, vyold, vzold``, bohr per atomic unit of time: the velocity
        estimate at ``current_positions``.
    ``accelerations``
        ``accxold, accyold, acczold``, bohr per squared atomic unit of time:
        the accelerations at ``previous_positions``.
    ``temperature``
        ``mol_dynamic%tempi``, kelvin: the target temperature of the next
        call (the cooling schedule changes it).
    ``frame``
        ``mol_dynamic%iframe``: number of the next time step; it continues
        across restarts.
    ``move_count``
        ``move%num``: number of calls made in this run.
    ``restarted``
        ``mol_dynamic%is_restart``: the run started from ``mdinit.dat``.
    """

    masses: np.ndarray
    movable: np.ndarray
    current_positions: np.ndarray
    previous_positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    temperature: float
    frame: int = 0
    move_count: int = 0
    restarted: bool = False

    def __post_init__(self) -> None:
        masses = np.array(self.masses, dtype=float)
        if masses.ndim != 1 or masses.size == 0:
            raise ValueError("masses must be a non-empty one-dimensional array")
        if not np.all(masses > 0.0):
            raise ValueError("masses must be positive")
        count = masses.size
        movable = np.array(self.movable)
        if movable.shape != (count,) or not np.all((movable == 0) | (movable == 1)):
            raise ValueError("movable must hold one 0 or 1 per atom")
        movable = movable.astype(np.int64)
        if int(movable.sum()) == 0:
            raise ValueError(
                "no atom is movable: PARSEC divides by the mass of the movable atoms"
            )
        masses.setflags(write=False)
        movable.setflags(write=False)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "movable", movable)
        for name in (
            "current_positions",
            "previous_positions",
            "velocities",
            "accelerations",
        ):
            object.__setattr__(
                self, name, _readonly(getattr(self, name), (count, 3), name)
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(self, "move_count", int(self.move_count))
        object.__setattr__(self, "restarted", bool(self.restarted))

    @property
    def atom_count(self) -> int:
        return int(self.masses.size)

    @property
    def is_first_step(self) -> bool:
        """The test of ``moldyn`` for drawing velocities: ``imove == 0`` and no restart."""
        return self.move_count == 0 and not self.restarted


@dataclass(frozen=True, eq=False)
class MoldynStep:
    """Result of one call of ``moldyn``.

    ``state`` is the state for the next call and ``positions`` (bohr, shape
    ``(atoms, 3)``) the new cluster coordinates ``clust%xatm``, at which the
    next self-consistent calculation is done.  The other fields are the
    line of ``md_nrg.dat`` in full precision:

    ``frame``
        ``iframe``, the number of the time step whose forces came in.
    ``time_ps``
        ``iframe * dt`` in picoseconds.
    ``kinetic_energy_ev``
        ``tkeev``: kinetic energy at ``frame`` per *movable* atom, eV.
    ``potential_energy_ev``
        ``bdev``: total energy of the electronic structure per atom, eV.
    ``total_energy_ev``
        their sum, the conserved quantity as PARSEC prints it.
    ``target_temperature``
        ``tempi`` used in this call, kelvin.
    ``temperature``
        ``tcalc``: temperature from the kinetic energy, kelvin.

    Beyond what PARSEC prints:

    ``kinetic_energy``
        total kinetic energy of the atoms at ``frame`` in Ry.
    ``conserved_energy``
        ``total_energy + kinetic_energy`` in Ry, the conserved quantity of
        the microcanonical run without the per-atom normalisations.
    ``center_of_mass_velocity``
        mass-weighted mean of the new velocity estimates of the movable
        atoms; ``moldyn`` computes it "for reporting" and never prints it.
    ``last_step``
        ``moldyn`` took its last-step branch: this was call ``Step_num`` and
        the final temperature is reached.  It then writes the coordinates
        "for next Run" to ``atom.cor`` and nothing to ``movie.xyz``.
    ``finished``
        the atom movement loop of ``parsec.F90`` ends after this call
        (``move%num > Step_num``); ``positions`` is then not used any more.
    """

    state: MolecularDynamicsState
    positions: np.ndarray
    frame: int
    time_ps: float
    kinetic_energy_ev: float
    potential_energy_ev: float
    total_energy_ev: float
    target_temperature: float
    temperature: float
    kinetic_energy: float
    conserved_energy: float
    center_of_mass_velocity: np.ndarray
    last_step: bool
    finished: bool


def start_molecular_dynamics(
    settings: MolecularDynamicsSettings,
    positions: np.ndarray | Sequence,
    masses: np.ndarray | Sequence,
    movable: np.ndarray | Sequence | None = None,
) -> MolecularDynamicsState:
    """State of a fresh run, as ``initial.f90`` prepares it.

    ``positions`` (bohr, ``(atoms, 3)``) are the cluster coordinates of the
    first self-consistent calculation, ``masses`` (electron masses) come from
    :func:`~parsec_python.MolecularDynamics.atomic_masses`, and ``movable``
    is ``clust%mvat`` (default: every atom moves).  Velocities and
    accelerations are set by the first call of :func:`moldyn_step`; they are
    zero here, where the Fortran arrays are still undefined.

    ``initial.f90`` shifts a cluster whose bounding box is centred more than
    0.1 bohr from the origin before it copies the coordinates into the
    molecular dynamics arrays (unless ``Restart_run`` is set); pass the
    shifted coordinates, for example those returned by
    ``parsec_python.V_ion.center_cluster_geometry``.
    """
    positions = np.array(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (atoms, 3)")
    count = positions.shape[0]
    if movable is None:
        movable = np.ones(count, dtype=np.int64)
    zeros = np.zeros((count, 3))
    return MolecularDynamicsState(
        masses=masses,
        movable=movable,
        current_positions=positions,
        previous_positions=zeros,
        velocities=zeros,
        accelerations=zeros,
        temperature=settings.initial_temperature,
        frame=0,
        move_count=0,
        restarted=False,
    )


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    """``dot_product`` as the Fortran loop adds it: from the first atom on."""
    return float(np.cumsum(left * right)[-1])


def _kinetic_energy(masses: np.ndarray, velocities: np.ndarray) -> float:
    """``tke = tke + half*amass(j)*(vx*vx + vy*vy + vz*vz)`` over the atoms."""
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    vz = velocities[:, 2]
    return float(np.cumsum(0.5 * masses * (vx * vx + vy * vy + vz * vz))[-1])


def _center_of_mass_velocity(
    masses: np.ndarray, amove: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    tmobilemass = _dot(masses, amove)
    return np.array(
        [_dot(masses, velocities[:, axis]) / tmobilemass for axis in range(3)]
    )


def _gaussian_deviates(count: int, random: UniformSource | None) -> np.ndarray:
    """``gdev`` for x, y and z of the first atom, then of the second, ..."""
    if random is None:
        raise ValueError(
            "this step draws random numbers: pass random=settings.random_source()"
        )
    deviates = np.empty((count, 3))
    for atom in range(count):
        for axis in range(3):
            deviates[atom, axis] = gaussian_deviate(random)
    return deviates


def moldyn_step(
    settings: MolecularDynamicsSettings,
    state: MolecularDynamicsState,
    forces: np.ndarray | Sequence,
    total_energy: float,
    *,
    random: UniformSource | None = None,
) -> MoldynStep:
    """Advance the atoms by one time step (one call of ``moldyn``).

    Parameters
    ----------
    settings:
        the molecular dynamics input.
    state:
        result of :func:`start_molecular_dynamics`, of
        :func:`~parsec_python.MolecularDynamics.restart_molecular_dynamics`,
        or ``state`` of the previous step.  It is not modified.
    forces:
        ``clust%force`` at ``state.current_positions``, shape ``(atoms, 3)``,
        Ry/bohr.  Forces on fixed atoms are ignored.
    total_energy:
        ``elec_st%etot``, the total energy of that structure in Ry.  PARSEC
        hands ``moldyn`` the energy per atom in eV,
        ``bdev = 13.6058*etot/atoms``, which is formed here.
    random:
        source of uniform random numbers, see
        :meth:`MolecularDynamicsSettings.random_source`.  It is needed by the
        first call of a fresh run (initial velocities) and by every call of a
        Langevin run; it is advanced in place.

    The caller ends the run when ``finished`` is true, and does not call at
    all when ``settings.step_num`` is zero, as ``parsec.F90`` does.
    """
    if settings.step_num == 0:
        raise ValueError("Step_num is zero: PARSEC does not call moldyn at all")
    count = state.atom_count
    forces = np.array(forces, dtype=float)
    if forces.shape != (count, 3):
        raise ValueError(f"forces must have shape ({count}, 3)")

    amass = state.masses[:, None]
    amove1 = state.movable.astype(float)
    amove = amove1[:, None]
    nmobileatom = int(state.movable.sum())
    tempi = state.temperature
    tempf = settings.final_temperature
    beta = settings.friction_coefficient
    deltat = settings.time_step
    iframe = state.frame
    imove = state.move_count
    xcur = state.current_positions

    # temperature in Hartree
    tempeh = tempi * KELVIN_RYDBERG * 0.5

    if state.is_first_step:
        # Maxwell velocities, 0.5 k T = 0.5 m vel**2 per component
        vel = np.sqrt(tempeh / amass)
        vold = vel * _gaussian_deviates(count, random) * amove
        # hold the centre of mass of the movable atoms
        vav = _center_of_mass_velocity(state.masses, amove1, vold)
        vold = vold - vav * amove
        tke = _kinetic_energy(state.masses, vold)
        # rescale to the input temperature
        if nmobileatom > 1:
            if beta < RESCALE_FRICTION:
                rescale = 2.0 * tke / float(3 * nmobileatom)
            else:
                rescale = 2.0 * tke / float(3 * nmobileatom - 3)
            rescale = math.sqrt(tempeh / rescale)
        else:
            rescale = 1.0
        vold = vold * rescale
        tke = tke * rescale * rescale

        xold = xcur
        if beta < MICROCANONICAL_FRICTION:
            accold = 0.5 * forces / amass * amove
            xcur = xold + deltat * (vold + 0.5 * accold * deltat)
            vold = vold + accold * deltat
        else:
            sigm = np.sqrt(2.0 * beta * tempeh / (deltat * amass))
            accold = (
                0.5 * forces / amass
                - beta * vold
                + sigm * _gaussian_deviates(count, random)
            ) * amove
            xcur = xold + deltat * (vold + 0.5 * deltat * accold)
            vold = vold + deltat * accold
    elif beta < MICROCANONICAL_FRICTION:
        accolder = state.accelerations
        # velocity at the current positions: 2 a = F/m in Hartree units
        vold = amove * (
            (xcur - state.previous_positions) / deltat
            + deltat * (forces / amass + accolder) / 6.0
        )
        vav = _center_of_mass_velocity(state.masses, amove1, vold)
        vold = vold - vav * amove
        xold = xcur
        xcur = xold + deltat * (
            vold + deltat * (2.0 * forces * amove / amass - accolder) / 6.0
        )
        tke = _kinetic_energy(state.masses, vold)
        vold = vold + deltat * (1.5 * forces * amove / amass - accolder) * 0.5
        accold = 0.5 * forces * amove / amass
    else:
        sigm = np.sqrt(2.0 * beta * tempeh / (deltat * amass))
        xolder = state.previous_positions
        xold = xcur
        accolder = state.accelerations
        # the previous velocity estimate enters the friction force
        accold = amove * (
            0.5 * forces / amass
            - beta * state.velocities
            + sigm * _gaussian_deviates(count, random)
        )
        voldr = (xold - xolder) / deltat + deltat * (2.0 * accold + accolder) / 6.0
        xcur = xold + deltat * (voldr + deltat * (4.0 * accold - accolder) / 6.0)
        tke = _kinetic_energy(state.masses, voldr)
        vold = voldr + deltat * (3.0 * accold - accolder) * 0.5

    # "just for the purpose of reporting"
    vav = _center_of_mass_velocity(state.masses, amove1, vold)

    with np.errstate(divide="ignore", invalid="ignore"):
        mobile = np.float64(nmobileatom)
        # kinetic energy per movable atom, Ry and eV
        tke_atom = np.float64(tke) * 2.0 / mobile
        tkeev = tke_atom * RYDBERG_EV
        if beta < MICROCANONICAL_FRICTION:
            tcalc = 2.0 / 3.0 * tke_atom / KELVIN_RYDBERG
        else:
            tcalc = 2.0 / 3.0 * tke_atom / KELVIN_RYDBERG * mobile / (mobile - 1.0)
    # totnrg.f90: total energy per atom in eV
    bdev = RYDBERG_EV * float(total_energy) / float(count)

    last_step = not (imove != settings.step_num or abs(tempi - tempf) > 1.0e-5)

    # temperature of the next call
    new_tempi = tempi
    tscale = settings.temperature_scale
    if settings.cooling_method == "stair":
        if (imove + 1) % settings.stride == 0:
            if abs(new_tempi - tempf) > 1.0e-5:
                new_tempi = new_tempi + tscale
                # do not go past the final temperature
                if (new_tempi - tempf) * tscale > 0.0:
                    new_tempi = tempf
    elif settings.cooling_method == "linear":
        new_tempi = new_tempi + tscale
    else:
        new_tempi = new_tempi * tscale

    new_state = replace(
        state,
        current_positions=xcur,
        previous_positions=xold,
        velocities=vold,
        accelerations=accold,
        temperature=new_tempi,
        frame=iframe + 1,
        move_count=imove + 1,
    )
    kinetic_energy = 2.0 * tke
    return MoldynStep(
        state=new_state,
        positions=new_state.current_positions,
        frame=iframe,
        time_ps=float(iframe) * deltat * TIME_AU_TO_PS,
        kinetic_energy_ev=float(tkeev),
        potential_energy_ev=bdev,
        total_energy_ev=float(tkeev) + bdev,
        target_temperature=tempi,
        temperature=float(tcalc),
        kinetic_energy=kinetic_energy,
        conserved_energy=float(total_energy) + kinetic_energy,
        center_of_mass_velocity=vav,
        last_step=last_step,
        finished=imove + 1 > settings.step_num,
    )
