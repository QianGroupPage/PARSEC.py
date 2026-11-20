"""``Remove_COM_Motion`` and ``MD_Append_Output`` of the molecular dynamics (PARSEC.py extensions).

* a Langevin run whose noise repeats (``Rng_Seed`` > 0: PARSEC puts the seed again every 10000 numbers, and every
  restart starts again from it) pushes the centre of mass the same way again and again; with ``Remove_COM_Motion``
  the random force has no net part and a restart takes the drift out of ``mdinit.dat``;
* a restart with ``MD_Append_Output`` goes on in the files of the run it continues: the records from the restart
  step on are cut, the headers are not written again, the step counter goes on.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.MolecularDynamics import MolecularDynamicsSettings
from parsec_python.MolecularDynamics.md_files import (
    last_md_mech_step,
    parse_md_restart,
    restart_molecular_dynamics,
    trim_output_file,
    trim_output_files,
)
from parsec_python.MolecularDynamics.moldyn import (
    moldyn_step,
    remove_center_of_mass_motion,
    start_molecular_dynamics,
)
from parsec_python.MolecularDynamics.run import run_recorded_molecular_dynamics
from parsec_python.Relaxation.backend import IonicBackend

MASS = 20.0 * 1822.9


def _langevin(**changes) -> MolecularDynamicsSettings:
    values = dict(
        enabled=True,
        friction_coefficient=1.2e-5,
        initial_temperature=1200.0,
        final_temperature=1200.0,
        time_step=41.34,
        step_num=250,
        rng_seed=3300,
    )
    values.update(changes)
    return MolecularDynamicsSettings(**values)


def _trap(positions0: np.ndarray):
    def force(positions: np.ndarray) -> np.ndarray:
        f = -0.02 * (positions - positions0)
        return f - f.mean(axis=0)

    return force


def _com_share(masses: np.ndarray, velocities: np.ndarray) -> float:
    centre = (masses[:, None] * velocities).sum(axis=0) / masses.sum()
    kinetic = 0.5 * (masses[:, None] * velocities**2).sum()
    return float(0.5 * masses.sum() * (centre**2).sum() / kinetic)


class KeywordTests(unittest.TestCase):
    def test_keywords_are_read(self) -> None:
        default = MolecularDynamicsSettings.from_keywords({"Molecular_Dynamics": ".true."})
        self.assertFalse(default.remove_center_of_mass)
        self.assertFalse(default.append_output)
        given = MolecularDynamicsSettings.from_keywords(
            {"Molecular_Dynamics": ".true.", "Remove_COM_Motion": ".true.", "MD_Append_Output": ""}
        )
        self.assertTrue(given.remove_center_of_mass)
        self.assertTrue(given.append_output)
        refused = MolecularDynamicsSettings.from_keywords(
            {"Molecular_Dynamics": ".true.", "remove_com_motion": "no"}
        )
        self.assertFalse(refused.remove_center_of_mass)


class CenterOfMassTests(unittest.TestCase):
    def _segments(self, remove: bool, reseed: bool, segments: int = 8) -> list[float]:
        rng = np.random.default_rng(1)
        count = 16
        positions0 = rng.uniform(0.0, 10.0, (count, 3))
        masses = np.full(count, MASS)
        force = _trap(positions0)
        settings = _langevin(remove_center_of_mass=remove)
        state = start_molecular_dynamics(settings, positions0, masses)
        shares = []
        for segment in range(segments):
            seed = 3300 + (segment if reseed else 0)
            source = replace(settings, rng_seed=seed).random_source(None, compiler="gfortran")
            for _ in range(250):
                state = moldyn_step(settings, state, force(state.current_positions), 0.0, random=source).state
            shares.append(_com_share(masses, state.velocities))
        return shares

    def test_repeating_noise_drives_the_centre_of_mass(self) -> None:
        drifting = self._segments(remove=False, reseed=False)
        self.assertGreater(max(drifting[4:]), 0.3)

    def test_option_keeps_the_centre_of_mass_at_rest(self) -> None:
        for reseed in (False, True):
            with self.subTest(reseed=reseed):
                self.assertLess(max(self._segments(remove=True, reseed=reseed, segments=3)), 1.0e-20)

    def test_default_is_unchanged(self) -> None:
        """Without the option the steps are those of PARSEC (the same numbers as before the keyword)."""
        rng = np.random.default_rng(2)
        positions0 = rng.uniform(0.0, 10.0, (6, 3))
        masses = np.full(6, MASS)
        force = _trap(positions0)
        plain = _langevin(step_num=20)
        explicit = replace(plain, remove_center_of_mass=False)
        states = []
        for settings in (plain, explicit):
            state = start_molecular_dynamics(settings, positions0, masses)
            source = settings.random_source(None, compiler="gfortran")
            for _ in range(20):
                state = moldyn_step(settings, state, force(state.current_positions), 0.0, random=source).state
            states.append(state)
        np.testing.assert_array_equal(states[0].current_positions, states[1].current_positions)

    def test_restart_takes_the_drift_out(self) -> None:
        rng = np.random.default_rng(3)
        count = 8
        positions0 = rng.uniform(0.0, 10.0, (count, 3))
        masses = np.full(count, MASS)
        masses[::2] *= 3.0
        force = _trap(positions0)
        settings = _langevin(step_num=300)
        state = start_molecular_dynamics(settings, positions0, masses)
        source = settings.random_source(None, compiler="gfortran")
        for _ in range(50):
            state = moldyn_step(settings, state, force(state.current_positions), 0.0, random=source).state
        # give the cell a drift, as the repeated noise did
        push = np.array([2.0e-4, -3.0e-4, 1.0e-4])
        drifting = replace(
            state,
            velocities=state.velocities + push,
            previous_positions=state.previous_positions - push * settings.time_step,
        )
        restart = SimpleNamespace(
            frame=drifting.frame,
            current_positions=drifting.current_positions,
            previous_positions=drifting.previous_positions,
            velocities=drifting.velocities,
            accelerations=drifting.accelerations,
            movable=np.ones(count),
        )
        plain = restart_molecular_dynamics(settings, restart, masses)
        cleaned = restart_molecular_dynamics(replace(settings, remove_center_of_mass=True), restart, masses)
        self.assertGreater(_com_share(masses, plain.velocities), 0.1)
        self.assertLess(_com_share(masses, cleaned.velocities), 1.0e-20)
        # the integrator sees no drift either: the next step moves the centre of mass by nothing
        with_option = replace(settings, remove_center_of_mass=True)
        step = moldyn_step(with_option, cleaned, force(cleaned.current_positions), 0.0, random=with_option.random_source(None, compiler="gfortran"))
        centre_before = (masses[:, None] * cleaned.current_positions).sum(axis=0) / masses.sum()
        centre_after = (masses[:, None] * step.positions).sum(axis=0) / masses.sum()
        np.testing.assert_allclose(centre_after, centre_before, atol=1.0e-9)
        # fixed atoms stay fixed
        movable = np.ones(count)
        movable[:2] = 0.0
        pinned = remove_center_of_mass_motion(replace(plain, movable=movable.astype(np.int64)), settings.time_step)
        np.testing.assert_array_equal(pinned.velocities[:2], plain.velocities[:2])


class TrimTests(unittest.TestCase):
    def _files(self, directory: Path) -> None:
        (directory / "md_nrg.dat").write_text(
            "Step Time\n" + "".join(f"{k:6d} {0.001 * k:9.3f} -1.0\n" for k in range(1, 8))
        )
        (directory / "md_mech.dat").write_text(
            "Each line\n" + "".join(f"Step #{k:6d}\n  1.0\n  2.0\n" for k in (10, 20, 30))
        )
        (directory / "atom.cor").write_text(
            "".join(f" Coordinates for step #{k:10d}\n Li 1.0 2.0 3.0\n" for k in (10, 20, 30))
            + "\n  Coordinates for next Run \n  1.0  2.0  3.0\n"
        )
        (directory / "movie.xyz").write_text(
            "".join(f"{1:7d}\n Step # {k:11d}\nLi    1.0   2.0   3.0\n" for k in (10, 20, 30))
        )
        (directory / "md_frames.extxyz").write_text(
            "".join(f'1\nLattice="1 0 0 0 1 0 0 0 1" energy=-1.0 step={k}\nLi 0.0 0.0 0.0 0.1 0.2 0.3\n' for k in (10, 20, 30))
        )

    def test_records_from_the_restart_step_on_are_cut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._files(directory)
            removed = trim_output_files(directory, 20)
            self.assertEqual(removed["md_nrg.dat"], 0)  # its rows are 1-7
            self.assertTrue(all(count > 0 for name, count in removed.items() if name != "md_nrg.dat"), removed)
            nrg = (directory / "md_nrg.dat").read_text().splitlines()
            self.assertEqual(nrg[0], "Step Time")
            self.assertEqual([int(line.split()[0]) for line in nrg[1:]], [1, 2, 3, 4, 5, 6, 7])  # every row is below 20
            self.assertEqual((directory / "md_mech.dat").read_text().count("Step #"), 1)
            self.assertEqual((directory / "atom.cor").read_text().count("Coordinates for step"), 1)
            self.assertNotIn("next Run", (directory / "atom.cor").read_text())
            self.assertEqual((directory / "movie.xyz").read_text().count("Step #"), 1)
            self.assertEqual((directory / "md_frames.extxyz").read_text().count("step="), 1)

    def test_md_nrg_rows_and_nothing_to_cut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._files(directory)
            self.assertEqual(trim_output_file(directory / "md_nrg.dat", 5), len("     5     0.005 -1.0\n     6     0.006 -1.0\n     7     0.007 -1.0\n"))
            self.assertEqual(
                [int(line.split()[0]) for line in (directory / "md_nrg.dat").read_text().splitlines()[1:]], [1, 2, 3, 4]
            )
            self.assertEqual(trim_output_file(directory / "md_nrg.dat", 99), 0)
            self.assertEqual(trim_output_file(directory / "movie.xyz", 99), 0)
            self.assertEqual(trim_output_file(directory / "md_mech.dat", 31), 0)


class _Backend:
    """A stand-in for the electronic structure: harmonic forces of a few atoms, net force removed."""

    def __init__(self, positions0: np.ndarray) -> None:
        self.positions0 = positions0

    def run_scf(self, system, callback=None, warm_start=None, **_):
        positions = np.array([atom.position for atom in system.atoms], dtype=float)
        energy = float(0.01 * np.sum((positions - self.positions0) ** 2))
        return SimpleNamespace(energies=SimpleNamespace(total=energy), converged=True, iterations=1, positions=positions)

    def compute_forces(self, system, result, *, symmetrize=True, operations=None):
        force = -0.02 * (result.positions - self.positions0)
        return SimpleNamespace(total=force - force.mean(axis=0), maximum=1.0)

    def update_geometry(self, system, atoms):
        return SimpleNamespace(atoms=tuple(atoms), input=system.input)

    def warm_start(self, previous_system, previous_result, system):
        return None

    def backend(self) -> IonicBackend:
        return IonicBackend(
            run_scf=self.run_scf,
            compute_forces=self.compute_forces,
            update_geometry=self.update_geometry,
            warm_start=self.warm_start,
        )


def _translation(settings: MolecularDynamicsSettings, interval: int = 2):
    return SimpleNamespace(
        molecular_dynamics=settings,
        movable=(True,) * 3,
        potential_extrapolation="none",
        md_output_interval=interval,
        problem=SimpleNamespace(
            eigensolver=SimpleNamespace(filter_restart=(0, 0, 0)),
            pseudopotentials={"H": SimpleNamespace(element_symbol="H")},
            atoms=None,
        ),
    )


def _system(positions: np.ndarray):
    from parsec_python.models import Atom

    atoms = tuple(Atom("H", np.array(row, dtype=float)) for row in positions)
    return SimpleNamespace(atoms=atoms, input=SimpleNamespace(eigensolver=SimpleNamespace(filter_restart=(0, 0, 0))))


def _rows(path: Path) -> list[int]:
    return [int(line.split()[0]) for line in path.read_text().splitlines()[1:] if line.split()[:1] and line.split()[0].isdigit()]


class AppendTests(unittest.TestCase):
    positions = np.array([[0.0, 0.0, 0.7], [0.0, 1.9, -0.7], [1.5, 0.3, 0.1]])

    def _run(self, directory: Path, settings, restart=None, system_positions=None):
        fake = _Backend(self.positions)
        reporter = SimpleNamespace(iteration=lambda *_: None, finish=lambda *_, **__: None, frame_detail="full")
        lines: list[str] = []
        outcome = run_recorded_molecular_dynamics(
            _translation(settings),
            _system(self.positions if system_positions is None else system_positions),
            fake.backend(),
            reporter,
            lines.append,
            directory=directory,
            restart=restart,
            elements=["H"] * 3,
        )
        return outcome, lines

    def test_restart_goes_on_in_the_same_files(self) -> None:
        settings = _langevin(step_num=6, use_test_rng=True, rng_seed=7, initial_temperature=300.0, final_temperature=300.0)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first, _ = self._run(directory, replace(settings, step_num=6, append_output=True, remove_center_of_mass=True))
            nrg_first = _rows(directory / "md_nrg.dat")
            self.assertEqual(nrg_first, [0, 1, 2, 3, 4, 5, 6][: len(nrg_first)])
            # a restart at an earlier block: whatever the first run wrote from there on is cut
            text = (directory / "md_mech.dat").read_text()
            blocks = [i for i, line in enumerate(text.splitlines()) if line.strip().startswith("Step #")]
            self.assertGreaterEqual(len(blocks), 3)
            lines = text.splitlines()
            start = blocks[1]
            restart_text = "\n".join(lines[start : start + 1 + 3]) + "\n"
            restart = parse_md_restart(restart_text, 3)
            (directory / "mdinit.dat").write_text(restart_text)
            restart_settings = replace(settings, restart=True, step_num=5, append_output=True, remove_center_of_mass=True)
            _, out = self._run(directory, restart_settings, restart=restart, system_positions=restart.current_positions)
            self.assertTrue(any("MD_Append_Output" in line for line in out), out)
            rows = _rows(directory / "md_nrg.dat")
            self.assertEqual(rows, list(range(rows[0], rows[0] + len(rows))))  # one run of frames: no repeat, no gap
            self.assertEqual(rows[-1], restart.frame + 5)
            self.assertEqual((directory / "md_nrg.dat").read_text().count("Step"), 1)  # one header
            mech = (directory / "md_mech.dat").read_text()
            self.assertEqual(mech.count("Each line contains"), 1)
            labels = [int(line.split("#")[1]) for line in mech.splitlines() if line.strip().startswith("Step #")]
            self.assertEqual(labels, sorted(set(labels)))  # numbers only go up
            movie = (directory / "movie.xyz").read_text()
            frames = [int(line.split("#")[1]) for line in movie.splitlines() if line.strip().startswith("Step #")]
            self.assertEqual(frames, sorted(set(frames)))

    def test_without_the_option_a_restart_starts_the_files_anew(self) -> None:
        settings = _langevin(step_num=4, use_test_rng=True, rng_seed=7, initial_temperature=300.0, final_temperature=300.0)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._run(directory, settings)
            text = (directory / "md_mech.dat").read_text()
            lines = text.splitlines()
            blocks = [i for i, line in enumerate(lines) if line.strip().startswith("Step #")]
            restart_text = "\n".join(lines[blocks[1] : blocks[1] + 4]) + "\n"
            restart = parse_md_restart(restart_text, 3)
            self._run(directory, replace(settings, restart=True, step_num=2), restart=restart, system_positions=restart.current_positions)
            rows = _rows(directory / "md_nrg.dat")
            self.assertEqual(rows[0], restart.frame)
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
