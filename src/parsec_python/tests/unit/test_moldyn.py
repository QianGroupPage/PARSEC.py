"""Source-parity tests for the molecular dynamics of PARSEC.

The reference data in ``tests/data/moldyn_fortran_reference.json`` were
produced by the original Fortran: ``usrinputfile.F90``, ``initial.f90``,
``moldyn.f90`` and ``ptable.f90`` compiled unchanged with gfortran 13.3 and
driven by a small program that follows the atom movement loop of
``parsec.F90`` with an analytic Morse cluster (three Si, two H, one Si
fixed) in place of the self-consistent field.  Numbers were written with 17
significant digits, so they are the exact doubles of the Fortran run.  The
description inside the file states the two places where a copy of a Fortran
file had to differ by one line, and the file carries the driver programs
and the build script that produced it (``harness_sources``).

``tests/data/moldyn_parsec_executable_reference.json`` is a short run of
the PARSEC executable itself (a water molecule) and is used by
``ParsecExecutableTests``.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

import numpy as np

from parsec_python.MolecularDynamics import (
    MD_MECH_HEADER,
    MD_NRG_HEADER,
    PARSEC_ATOMIC_WEIGHTS,
    BufferedRandom,
    GfortranRandom,
    MolecularDynamicsInputError,
    MolecularDynamicsSettings,
    MolecularDynamicsState,
    ParsecTestRandom,
    atomic_masses,
    atomic_weight,
    check_run_mode,
    format_atom_cor_step,
    format_md_mech_step,
    format_md_nrg_line,
    format_md_restart,
    format_movie_xyz_frame,
    format_parsec_out_summary,
    gaussian_deviate,
    initial_records,
    last_md_mech_step,
    moldyn_step,
    parse_md_restart,
    restart_molecular_dynamics,
    run_molecular_dynamics,
    start_molecular_dynamics,
    step_records,
)
from parsec_python.MolecularDynamics.fortran_format import (
    fortran_e,
    fortran_f,
    fortran_i,
    list_directed_integer,
    read_fortran_integer,
    read_fortran_real,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "moldyn_fortran_reference.json"
REFERENCE = json.loads(DATA.read_text(encoding="utf-8"))
CASES = REFERENCE["cases"]

# Cases whose input PARSEC accepts and that run molecular dynamics.
RUNNING = [
    name
    for name, case in CASES.items()
    if case.get("is_on") and case.get("moldyn_calls", 0) > 0
]
# Runs in which every random number comes from PARSEC's own generator and
# can therefore be repeated here, including the first step.
REPEATABLE = [name for name in RUNNING if CASES[name]["settings"]["use_test_rng"]]
NEWTONIAN = [
    name for name in RUNNING if CASES[name]["settings"]["friction_coef"] < 1.0e-6
]
LANGEVIN = [name for name in RUNNING if name not in NEWTONIAN]

# A trajectory is compared with this relative tolerance.  On a platform
# whose C library rounds log() like the Fortran run-time the agreement is
# exact; the tolerance leaves room for a last-bit difference in a Gaussian
# deviate.
TRAJECTORY_RTOL = 1.0e-12


def _number(value) -> float:
    """Numbers of the reference; infinity and NaN are stored as strings."""
    return float(value)


class _Case:
    """One Fortran run of the reference file with the Python objects for it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.data = CASES[name]
        self.settings = MolecularDynamicsSettings.from_keywords(self.data["keywords"])
        atoms = self.data["atoms"]
        self.species = [atom["name"].strip() for atom in atoms]
        self.masses = atomic_masses(self.species)
        self.movable = np.array([atom["mvat"] for atom in atoms])
        # clust%xatm after initial.f90: the structure of the first SCF
        self.first_positions = np.array([atom["position"] for atom in atoms])
        self.steps = [step for step in self.data["steps"] if "after" in step]

    def initial_state(self) -> MolecularDynamicsState:
        if self.settings.restart:
            restart = parse_md_restart(self.data["mdinit_dat"], len(self.species))
            return restart_molecular_dynamics(self.settings, restart, self.masses)
        return start_molecular_dynamics(
            self.settings, self.first_positions, self.masses, self.movable
        )

    def positions_before(self, index: int) -> np.ndarray:
        """xcur before call ``index``."""
        if index == 0:
            return np.array(self.data["state0"]["xcur"])
        return np.array(self.steps[index - 1]["after"]["xcur"])

    def state_before(self, index: int) -> MolecularDynamicsState:
        """The Fortran state that went into call ``index``."""
        if index == 0:
            return self.initial_state()
        after = self.steps[index - 1]["after"]
        return MolecularDynamicsState(
            masses=self.masses,
            movable=self.movable,
            current_positions=after["xcur"],
            previous_positions=self.positions_before(index - 1),
            velocities=after["vold"],
            accelerations=after["accold"],
            temperature=after["tempi"],
            frame=after["iframe"],
            move_count=after["imove"],
            restarted=self.settings.restart,
        )

    def seed_before(self, index: int) -> int:
        if index == 0:
            return self.settings.rng_seed
        return self.steps[index - 1]["after"]["rng_seed"]

    def draws_random_numbers(self, index: int) -> bool:
        first = index == 0 and not self.settings.restart
        return first or not self.settings.is_microcanonical


def _morse(positions: np.ndarray, species: list[str]) -> tuple[float, np.ndarray]:
    """The Morse cluster of the Fortran driver, same loops and same order."""
    depth = {"SiSi": 0.20, "SiH": 0.30, "HSi": 0.30, "HH": 0.05}
    alpha = {"SiSi": 0.90, "SiH": 1.10, "HSi": 1.10, "HH": 1.30}
    radius = {"SiSi": 4.20, "SiH": 2.90, "HSi": 2.90, "HH": 3.40}
    count = len(species)
    force = np.zeros((count, 3))
    energy = 0.0
    for i in range(count - 1):
        for j in range(i + 1, count):
            pair = species[i] + species[j]
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            ex = math.exp(-alpha[pair] * (r - radius[pair]))
            energy = energy + depth[pair] * ((1.0 - ex) * (1.0 - ex) - 1.0)
            de = 2.0 * depth[pair] * alpha[pair] * (1.0 - ex) * ex
            fr = -de / r
            force[i] += (fr * dx, fr * dy, fr * dz)
            force[j] -= (fr * dx, fr * dy, fr * dz)
    return energy, force


class FortranFormatTests(unittest.TestCase):
    def test_output_descriptors_match_gfortran(self) -> None:
        """Every F, E and I field of the files against the Fortran library."""
        real = {
            "F7.3": lambda v: fortran_f(v, 7, 3),
            "F7.2": lambda v: fortran_f(v, 7, 2),
            "F11.5": lambda v: fortran_f(v, 11, 5),
            "F11.6": lambda v: fortran_f(v, 11, 6),
            "F14.9": lambda v: fortran_f(v, 14, 9),
            "F10.6": lambda v: fortran_f(v, 10, 6),
            "E15.8": lambda v: fortran_e(v, 15, 8),
        }
        integer = {
            "I4": lambda v: fortran_i(v, 4),
            "I3": lambda v: fortran_i(v, 3),
            "LIST": lambda v: " Coordinates for step #" + list_directed_integer(v),
        }
        checked = 0
        for line in REFERENCE["format_probe"]:
            tag = line[:7].strip()
            if tag in real:
                value = float(line[7:32])
                self.assertEqual(real[tag](value), line[34:-1], line)
                checked += 1
            elif tag in integer:
                value = int(line[7:19])
                self.assertEqual(integer[tag](value), line[21:-1], line)
                checked += 1
        self.assertEqual(checked, 46 * 7 + 12 * 3)

    def test_overflow_signed_zero_and_special_values(self) -> None:
        self.assertEqual(fortran_f(-1.0e-9, 7, 3), " -0.000")
        self.assertEqual(fortran_f(33524.0, 7, 2), "*******")
        self.assertEqual(fortran_f(float("inf"), 7, 2), "    Inf")
        self.assertEqual(fortran_f(float("-inf"), 11, 5), "  -Infinity")
        self.assertEqual(fortran_f(float("nan"), 7, 2), "    NaN")
        self.assertEqual(fortran_e(-0.0, 15, 8), "-0.00000000E+00")
        self.assertEqual(fortran_e(1.0e100, 15, 8), " 0.10000000+101")
        self.assertEqual(fortran_i(10000, 4), "****")
        self.assertEqual(fortran_i(-999, 4), "-999")

    def test_input_fields_follow_the_fortran_rules(self) -> None:
        """The buffers of the Fortran probe, read with e15.8."""
        buffers = {
            "READ1": " 0.12345678E+01",
            "READ2": " 0.10000000-099",
            "READ3": "-0.10000000+100",
            "READ4": "       12345678",
            "READ5": "   1.5         ",
            "READ6": " 0.1234D+02    ",
            "READ7": "   12  3       ",
        }
        seen = 0
        for line in REFERENCE["format_probe"]:
            tag = line[:7].strip()
            if tag in buffers:
                self.assertEqual(int(line[7:11]), 0)
                self.assertEqual(
                    read_fortran_real(buffers[tag], 8), float(line[11:]), tag
                )
                seen += 1
        self.assertEqual(seen, len(buffers))
        self.assertEqual(read_fortran_real("", 8), 0.0)
        self.assertEqual(read_fortran_integer("  12"), 12)
        self.assertEqual(read_fortran_integer("    "), 0)
        with self.assertRaises(ValueError):
            read_fortran_real("  abc  ", 8)
        with self.assertRaises(ValueError):
            read_fortran_integer("****")

    def test_values_survive_the_restart_format(self) -> None:
        for value in (0.0, 1.0, -2.5e-7, 123.456789, -9.87654321e-12, 3.0e120):
            text = fortran_e(value, 15, 8)
            self.assertEqual(len(text), 15)
            back = read_fortran_real(text, 8)
            self.assertEqual(fortran_e(back, 15, 8), text)
            # eight significant digits
            self.assertLessEqual(abs(back - value), 5.0e-8 * abs(value))


class AtomicMassTests(unittest.TestCase):
    def test_table_equals_ptable(self) -> None:
        """Every name of the Fortran ``if`` chain; the first match wins."""
        fortran: dict[str, float] = {}
        unknown: list[str] = []
        for name, weight, ierr in REFERENCE["ptable"]:
            if ierr == 0:
                fortran.setdefault(name.strip(), weight)
            else:
                self.assertEqual(ierr, 201)
                unknown.append(name)
        self.assertEqual(set(fortran), set(PARSEC_ATOMIC_WEIGHTS))
        for name, weight in fortran.items():
            self.assertEqual(atomic_weight(name), weight, name)
        for name in unknown:
            with self.assertRaises(KeyError):
                atomic_weight(name)

    def test_slips_of_the_fortran_table_are_kept(self) -> None:
        self.assertEqual(atomic_weight("Pr"), 140.90765)
        with self.assertRaises(KeyError):
            atomic_weight("Pm")
        self.assertEqual(atomic_weight("F3"), atomic_weight("H"))
        self.assertEqual(atomic_weight("uF"), atomic_weight("Fe"))

    def test_masses_are_weights_times_1836(self) -> None:
        case = CASES["nve_stair"]
        names = [atom["name"] for atom in case["atoms"]]
        masses = atomic_masses(names)
        self.assertTrue(
            np.array_equal(masses, [atom["amass"] for atom in case["atoms"]])
        )
        self.assertEqual(masses[0], 28.0855 * 1836.0)
        self.assertEqual(masses[-1], 1.00794 * 1836.0)
        # a longer name is cut to the two characters of clust%name
        self.assertEqual(atomic_weight("Silicon"), atomic_weight("Si"))


class RandomNumberTests(unittest.TestCase):
    def test_uniform_numbers_equal_get_rand_test(self) -> None:
        for probe in REFERENCE["random"]["probes"]:
            source = ParsecTestRandom(probe["seed"])
            values = [source.uniform() for _ in probe["uniform"]]
            self.assertEqual(values, probe["uniform"], probe["seed"])
            self.assertEqual(source.seed, probe["uniform_final_seed"])

    def test_minimal_standard_generator_check_value(self) -> None:
        source = ParsecTestRandom(1)
        for _ in range(10000):
            source.uniform()
        self.assertEqual(source.seed, 1043618065)
        self.assertEqual(REFERENCE["random"]["seed_after_10000_calls_from_1"], 1043618065)

    def test_gaussian_deviates_equal_gdev(self) -> None:
        for probe in REFERENCE["random"]["probes"]:
            source = ParsecTestRandom(probe["seed"])
            values = [gaussian_deviate(source) for _ in probe["gaussian"]]
            # log() is the only operation that is not exactly rounded
            np.testing.assert_allclose(values, probe["gaussian"], rtol=1.0e-14, atol=0.0)
            self.assertEqual(source.seed, probe["gaussian_final_seed"])

    def test_gdev_returns_the_second_deviate_and_rejects_points(self) -> None:
        class Scripted:
            def __init__(self, numbers, rejects_small_radius):
                self.numbers = list(numbers)
                self.rejects_small_radius = rejects_small_radius

            def uniform(self):
                return self.numbers.pop(0)

        # (0.95, 0.95) lies outside the unit circle and is drawn again
        source = Scripted([0.95, 0.95, 0.75, 0.25], True)
        v1, v2 = 0.5, -0.5
        r = v1 * v1 + v2 * v2
        self.assertEqual(
            gaussian_deviate(source), v2 * math.sqrt(-2.0 * math.log(r) / r)
        )
        self.assertEqual(source.numbers, [])
        # a point within 1e-7 of the origin is rejected by the test generator
        tiny = 0.5 + 1.0e-5
        source = Scripted([tiny, tiny, 0.75, 0.25], True)
        gaussian_deviate(source)
        self.assertEqual(source.numbers, [])
        # ... and accepted with the compiler generator
        source = Scripted([tiny, tiny, 0.75, 0.25], False)
        gaussian_deviate(source)
        self.assertEqual(source.numbers, [0.75, 0.25])

    def test_buffered_source_follows_getrandom(self) -> None:
        # no rng_seed: the generator simply continues across refills
        source = BufferedRandom(np.random.default_rng(5), size=4)
        values = [source.uniform() for _ in range(10)]
        expected = np.random.default_rng(5)
        blocks = np.concatenate([expected.random(4) for _ in range(3)])
        self.assertEqual(values, list(blocks[:10]))
        self.assertEqual(BufferedRandom(5, size=4).uniform(), values[0])
        # rng_seed > 0: PARSEC puts the same seed before every refill
        source = BufferedRandom(rng_seed=17, size=4)
        values = [source.uniform() for _ in range(12)]
        self.assertEqual(values[0:4], values[4:8])
        self.assertEqual(values[0:4], values[8:12])
        self.assertEqual(values[0:4], list(np.random.default_rng(17).random(4)))
        self.assertFalse(source.rejects_small_radius)

    def test_gfortran_source_equals_the_compiler_generator(self) -> None:
        """25000 numbers of ``getrandom`` in a gfortran build, per seed."""
        for fortran in REFERENCE["random"]["compiler_generator"]:
            with self.subTest(rng_seed=fortran["rng_seed"]):
                source = GfortranRandom(fortran["rng_seed"])
                values = []
                total = 0.0
                for _ in range(25000):
                    values.append(source.uniform())
                    total = total + values[-1]
                self.assertEqual(values[:12], fortran["first"])
                self.assertEqual(values[9997:10003], fortran["around_first_refill"])
                self.assertEqual(values[19997:20003], fortran["around_second_refill"])
                self.assertEqual(values[-3:], fortran["last"])
                self.assertEqual(total, fortran["sum_of_25000"])
                self.assertEqual(source.position, 5001)
                # PARSEC puts rng_seed again before every refill of the
                # buffer: the Fortran numbers repeat after 10000
                self.assertTrue(fortran["second_buffer_repeats_the_first"])
                self.assertEqual(values[10000:20000], values[:10000])
        with self.assertRaises(ValueError):
            GfortranRandom(0)

    def test_settings_choose_the_source(self) -> None:
        test = MolecularDynamicsSettings(enabled=True, use_test_rng=True, rng_seed=9)
        self.assertIsInstance(test.random_source(), ParsecTestRandom)
        self.assertEqual(test.random_source().seed, 9)
        default = MolecularDynamicsSettings(enabled=True)
        first = default.random_source(3).uniform()
        self.assertEqual(first, np.random.default_rng(3).random(10000)[0])
        seeded = MolecularDynamicsSettings(enabled=True, rng_seed=12)
        self.assertEqual(
            seeded.random_source(3).uniform(),
            np.random.default_rng(12).random(10000)[0],
        )


class SettingsTests(unittest.TestCase):
    def test_defaults_are_those_of_usrinput(self) -> None:
        settings = MolecularDynamicsSettings()
        self.assertFalse(settings.enabled)
        self.assertFalse(settings.restart)
        self.assertEqual(settings.cooling_method, "stair")
        self.assertEqual(settings.initial_temperature, 500.0)
        self.assertEqual(settings.final_temperature, 500.0)
        self.assertEqual(settings.temperature_step, 150.0)
        self.assertEqual(settings.time_step, 150.0)
        self.assertEqual(settings.friction_coefficient, 0.0)
        self.assertEqual(settings.step_num, 250)
        self.assertEqual(settings.rng_seed, 0)
        self.assertFalse(settings.use_test_rng)
        self.assertEqual(MolecularDynamicsSettings.from_keywords({}), settings)
        # the Fortran run with nothing but Molecular_Dynamics: .true.
        fortran = CASES["defaults"]["settings"]
        enabled = MolecularDynamicsSettings.from_keywords(CASES["defaults"]["keywords"])
        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.step_num, fortran["step_num"])
        self.assertEqual(enabled.time_step, fortran["time_step"])
        self.assertEqual(enabled.stride, fortran["stride"])

    def test_every_input_is_read_as_usrinput_reads_it(self) -> None:
        for name, case in CASES.items():
            if not case.get("is_on"):
                continue
            with self.subTest(case=name):
                fortran = case["settings"]
                settings = MolecularDynamicsSettings.from_keywords(case["keywords"])
                self.assertTrue(settings.enabled)
                self.assertEqual(settings.cool_type, fortran["cool_type"])
                self.assertEqual(settings.step_num, fortran["step_num"])
                self.assertEqual(settings.stride, fortran["stride"])
                self.assertEqual(settings.restart, fortran["is_restart"])
                self.assertEqual(settings.use_test_rng, fortran["use_test_rng"])
                self.assertEqual(settings.rng_seed, fortran["rng_seed"])
                self.assertEqual(settings.initial_temperature, fortran["tempi"])
                self.assertEqual(settings.final_temperature, fortran["tempf"])
                self.assertEqual(settings.temperature_step, fortran["step_tmp"])
                self.assertEqual(settings.time_step, fortran["time_step"])
                self.assertEqual(
                    settings.friction_coefficient, fortran["friction_coef"]
                )
                if settings.cooling_method == "log":
                    # a power: the last bit depends on the C library
                    self.assertAlmostEqual(
                        settings.temperature_scale / fortran["tscale"], 1.0, delta=4.0e-16
                    )
                else:
                    self.assertEqual(settings.temperature_scale, fortran["tscale"])

    def test_units_follow_esdf(self) -> None:
        case = CASES["units_and_spelling"]
        settings = MolecularDynamicsSettings.from_keywords(case["keywords"])
        # the kelvin is an energy unit of ESDF: 0.05 eV and 2 mRy
        self.assertEqual(settings.initial_temperature, (1.60219e-19 / 1.38066e-23) * 0.05)
        self.assertEqual(settings.temperature_step, (2.17991e-21 / 1.38066e-23) * 2.0)
        self.assertEqual(settings.final_temperature, 300.0)
        # "2.5 fs" is converted to ps and that number is used as atomic units
        self.assertEqual(settings.time_step, (1.0e-15 / 1.0e-12) * 2.5)
        self.assertEqual(settings.time_step, case["settings"]["time_step"])
        self.assertEqual(settings.friction_coefficient, 1.0e-3)
        bare = MolecularDynamicsSettings.from_keywords({"time_step": "150"})
        self.assertEqual(bare.time_step, 150.0)
        self.assertAlmostEqual(bare.time_step_ps, 150.0 * 2.4188843e-5, places=15)
        for unit, factor in (("Ry", 2.17991e-18), ("hartree", 4.35982e-18),
                             ("cm-1", 1.986e-23), ("K", 1.38066e-23),
                             ("kcal/mol", 6.94780e-21)):
            settings = MolecularDynamicsSettings.from_keywords({"tinit": f"3.0 {unit}"})
            self.assertEqual(
                settings.initial_temperature, (factor / 1.38066e-23) * 3.0, unit
            )

    def test_lines_esdf_cannot_read_are_errors(self) -> None:
        bad = (
            {"tinit": "300 fs"},  # dimensions do not match
            {"time_step": "2 K"},
            {"tinit": "300 furlong"},  # unit not recognised
            {"tinit": "warm"},
            {"step_num": "5.0"},
            {"step_num": "many"},
            {"friction_coefficient": "weak"},
            {"molecular_dynamics": "1"},
            {"test_rng": "maybe"},
        )
        for keywords in bad:
            with self.subTest(keywords=keywords):
                with self.assertRaises(MolecularDynamicsInputError) as raised:
                    MolecularDynamicsSettings.from_keywords(keywords)
                self.assertIsNone(raised.exception.parsec_error_code)

    def test_keyword_spelling_and_logicals(self) -> None:
        settings = MolecularDynamicsSettings.from_keywords(
            {
                "MOLECULAR_DYNAMICS": "",  # a bare logical keyword is true
                "T.I-nit": "250",  # ESDF drops '.' and '-' and the case
                "t_final": "250",
                "Restart_Mode": "yes",
                "Test_RNG": ".T.",
                "Cooling_Method": "= LOG",
                "Friction_Coefficient": "2.5D-3",
                "unrelated_keyword": "ignored",
            }
        )
        self.assertTrue(settings.enabled)
        self.assertTrue(settings.restart)
        self.assertTrue(settings.use_test_rng)
        self.assertEqual(settings.cooling_method, "log")
        self.assertEqual(settings.initial_temperature, 250.0)
        self.assertEqual(settings.friction_coefficient, 2.5e-3)
        # the underscore is significant: 'T_init' is not the keyword 'Tinit'
        other = MolecularDynamicsSettings.from_keywords({"T_init": "250"})
        self.assertEqual(other.initial_temperature, 500.0)
        for word, value in (("no", False), (".false.", False), ("F", False),
                            ("TRUE", True), ("t", True)):
            self.assertEqual(
                MolecularDynamicsSettings.from_keywords(
                    {"molecular_dynamics": word}
                ).enabled,
                value,
            )
        # usrinput reads Restart_mode only when molecular dynamics is on
        off = MolecularDynamicsSettings.from_keywords({"restart_mode": ".true."})
        self.assertFalse(off.restart)

    def test_errors_of_the_fortran_program(self) -> None:
        expected = {
            "error_unknown_cooling": 154,
            "error_wrong_sign": 231,
            "error_wrong_sign_heating": 231,
            "error_step_too_large": 232,
            "error_md_and_minimization": 157,
        }
        for name, code in expected.items():
            with self.subTest(case=name):
                case = CASES[name]
                fortran = case["usrinput_ierr"] or case["initial_ierr"]
                self.assertEqual(fortran, code)
                keywords = dict(case["keywords"])
                if name == "error_md_and_minimization":
                    self.assertEqual(keywords["minimization"], "simple")
                with self.assertRaises(MolecularDynamicsInputError) as raised:
                    MolecularDynamicsSettings.from_keywords(keywords)
                self.assertEqual(raised.exception.parsec_error_code, code)
        # (Tinit - T_Final)/T_Step = -1 makes zero stairs: the Fortran program
        # dies of an integer division by zero in usrinput, before initial
        # can report the wrong sign.  Here the input gets that report.
        crash = CASES["error_wrong_sign_zero_stairs"]
        self.assertEqual(crash["fortran_crash"]["signal"], "SIGFPE")
        with self.assertRaises(MolecularDynamicsInputError) as raised:
            MolecularDynamicsSettings.from_keywords(crash["keywords"])
        self.assertEqual(raised.exception.parsec_error_code, 231)
        # the largest step PARSEC still accepts
        allowed = MolecularDynamicsSettings.from_keywords(
            CASES["stair_step_just_allowed"]["keywords"]
        )
        self.assertEqual(allowed.temperature_step, 199.0)
        # the cooling method is checked even when molecular dynamics is off
        with self.assertRaises(MolecularDynamicsInputError) as raised:
            MolecularDynamicsSettings(cooling_method="exponential")
        self.assertEqual(raised.exception.parsec_error_code, 154)
        # ... the stair is not
        MolecularDynamicsSettings(
            initial_temperature=600.0, final_temperature=300.0, temperature_step=-100.0
        )

    def test_run_modes_that_exclude_each_other(self) -> None:
        on = MolecularDynamicsSettings(enabled=True)
        off = MolecularDynamicsSettings()
        check_run_mode(on)
        check_run_mode(off, minimization="bfgs")
        with self.assertRaises(MolecularDynamicsInputError) as raised:
            check_run_mode(on, minimization="BFGS")
        self.assertEqual(raised.exception.parsec_error_code, 157)
        for settings, minimization in ((on, "none"), (off, "simple")):
            with self.assertRaises(MolecularDynamicsInputError) as raised:
                check_run_mode(settings, minimization=minimization, polarizability=True)
            self.assertEqual(raised.exception.parsec_error_code, 155)
        with self.assertRaises(MolecularDynamicsInputError) as raised:
            MolecularDynamicsSettings.from_keywords(
                {"molecular_dynamics": ".true.", "polarizability": ".true."}
            )
        self.assertEqual(raised.exception.parsec_error_code, 155)

    def test_md_switched_off_keeps_the_other_values(self) -> None:
        case = CASES["md_off"]
        self.assertFalse(case["is_on"])
        settings = MolecularDynamicsSettings.from_keywords(case["keywords"])
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.initial_temperature, 700.0)
        self.assertEqual(settings.step_num, 3)

    def test_values_the_fortran_program_would_turn_into_nonsense(self) -> None:
        bad = (
            dict(time_step=0.0),
            dict(time_step=-20.0),
            dict(step_num=-1),
            dict(initial_temperature=0.0, final_temperature=0.0),
            dict(initial_temperature=-5.0, final_temperature=-5.0),
            dict(final_temperature=float("nan")),
            dict(friction_coefficient=float("inf")),
            dict(restart=True, initial_temperature=0.0, cooling_method="log"),
        )
        for values in bad:
            with self.subTest(values=values):
                with self.assertRaises(MolecularDynamicsInputError) as raised:
                    MolecularDynamicsSettings(enabled=True, **values)
                self.assertIsNone(raised.exception.parsec_error_code)
        with self.assertRaises(MolecularDynamicsInputError):
            MolecularDynamicsSettings(step_num=2.5)
        # allowed: a restart at zero temperature, and a negative friction,
        # which PARSEC treats as "below 1e-6", that is microcanonical
        MolecularDynamicsSettings(
            enabled=True, restart=True, initial_temperature=0.0, final_temperature=0.0
        )
        self.assertTrue(
            MolecularDynamicsSettings(
                enabled=True, friction_coefficient=-1.0
            ).is_microcanonical
        )

    def test_stride_of_the_stair(self) -> None:
        def stride(tempi, tempf, step, count):
            return MolecularDynamicsSettings(
                enabled=True,
                initial_temperature=tempi,
                final_temperature=tempf,
                temperature_step=step,
                step_num=count,
            ).stride

        # i = int((600-350)/100 + 1) = 3 stairs, 6/3 = 2 steps each
        self.assertEqual(stride(600.0, 350.0, 100.0, 6), 2)
        # 7/3 = 2 with a remainder: one more
        self.assertEqual(stride(600.0, 350.0, 100.0, 7), 3)
        # fewer steps than stairs
        self.assertEqual(stride(600.0, 300.0, 100.0, 2), 1)
        # isothermal: one stair as long as the run
        self.assertEqual(stride(500.0, 500.0, 150.0, 250), 250)
        self.assertEqual(stride(500.0, 500.0, 0.0, 250), 1)
        self.assertEqual(MolecularDynamicsSettings(cooling_method="linear").stride, 0)


class TrajectoryTests(unittest.TestCase):
    def _compare_step(self, case, index, step, random, *, exact):
        """Compare one Python step with what Fortran wrote after call ``index``."""
        reference = case.steps[index]
        after = reference["after"]
        state = step.state
        pairs = (
            (state.current_positions, after["xcur"]),
            (state.previous_positions, case.positions_before(index)),
            (state.velocities, after["vold"]),
            (state.accelerations, after["accold"]),
        )
        energy = {key: _number(value) for key, value in reference["energy"].items()}
        mobile = float(case.movable.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            scalars = (
                # tke of the Fortran routine: Ry per movable atom
                (np.float64(step.kinetic_energy) / mobile, energy["tke"]),
                (step.kinetic_energy_ev, energy["tkeev"]),
                (step.temperature, energy["tcalc"]),
                (step.total_energy_ev, energy["etotev"]),
                (step.target_temperature, energy["tempi"]),
            )
        # The log schedule multiplies by a power whose last bit depends on
        # the C library; the other schedules are exact.
        if case.settings.cooling_method == "log":
            self.assertAlmostEqual(
                state.temperature / after["tempi"], 1.0, delta=1.0e-14
            )
        else:
            self.assertEqual(state.temperature, after["tempi"])
        for mine, theirs in pairs:
            if exact:
                self.assertTrue(np.array_equal(mine, np.array(theirs)), (case.name, index))
            else:
                np.testing.assert_allclose(
                    mine, theirs, rtol=TRAJECTORY_RTOL, atol=1.0e-300
                )
        for mine, theirs in scalars:
            if math.isnan(theirs):
                self.assertTrue(math.isnan(mine))
            elif exact or math.isinf(theirs) or theirs == 0.0:
                self.assertEqual(mine, theirs, (case.name, index))
            else:
                self.assertAlmostEqual(mine / theirs, 1.0, delta=TRAJECTORY_RTOL)
        self.assertEqual(step.potential_energy_ev, reference["bdev"])
        self.assertEqual(step.frame, reference["iframe"])
        self.assertEqual(state.frame, after["iframe"])
        self.assertEqual(state.move_count, after["imove"])
        if isinstance(random, ParsecTestRandom):
            self.assertEqual(random.seed, after["rng_seed"], (case.name, index))
        elif random is not None:
            # mol_dynamic%iset: how far into the buffer the run has got
            self.assertEqual(random.position, after["iset"], (case.name, index))

    def test_newtonian_steps_are_bitwise_identical(self) -> None:
        """Zero friction and given velocities: no rounding freedom at all.

        Every such step of every Fortran run is repeated from the Fortran
        state before it, including the runs whose first velocities came from
        the compiler's generator or from the system clock.
        """
        compared = 0
        for name in NEWTONIAN:
            case = _Case(name)
            for index, reference in enumerate(case.steps):
                if case.draws_random_numbers(index):
                    continue
                with self.subTest(case=name, step=index):
                    state = case.state_before(index)
                    step = moldyn_step(
                        case.settings, state, reference["forces"], reference["etot"]
                    )
                    self._compare_step(case, index, step, None, exact=True)
                    compared += 1
        self.assertGreaterEqual(compared, 40)

    def test_steps_with_random_numbers_agree_from_the_fortran_generator_state(self) -> None:
        compared = 0
        for name in REPEATABLE:
            case = _Case(name)
            for index, reference in enumerate(case.steps):
                if not case.draws_random_numbers(index):
                    continue
                with self.subTest(case=name, step=index):
                    random = ParsecTestRandom(case.seed_before(index))
                    step = moldyn_step(
                        case.settings,
                        case.state_before(index),
                        reference["forces"],
                        reference["etot"],
                        random=random,
                    )
                    self._compare_step(case, index, step, random, exact=False)
                    compared += 1
        self.assertGreaterEqual(compared, 40)

    def test_whole_runs_follow_fortran(self) -> None:
        """Python carries its own state through the run; Fortran gives the forces."""
        worst = 0.0
        for name in REPEATABLE:
            case = _Case(name)
            with self.subTest(case=name):
                state = case.initial_state()
                random = case.settings.random_source()
                for index, reference in enumerate(case.steps):
                    step = moldyn_step(
                        case.settings, state, reference["forces"], reference["etot"],
                        random=random,
                    )
                    self._compare_step(case, index, step, random, exact=False)
                    self.assertEqual(step.finished, index == len(case.steps) - 1)
                    after = reference["after"]
                    for mine, theirs in (
                        (step.state.current_positions, after["xcur"]),
                        (step.state.velocities, after["vold"]),
                    ):
                        theirs = np.array(theirs)
                        scale = np.where(theirs == 0.0, 1.0, np.abs(theirs))
                        worst = max(worst, float(np.max(np.abs(mine - theirs) / scale)))
                    state = step.state
                self.assertEqual(len(case.steps), case.settings.step_num + 1)
                self.assertEqual(len(case.steps), case.data["moldyn_calls"])
        self.assertLessEqual(worst, TRAJECTORY_RTOL)

    def test_runs_started_by_the_compiler_generator_continue_bitwise(self) -> None:
        """Given the velocities of the Fortran run, the rest is deterministic."""
        for name in ("nve_intrinsic_rng", "defaults"):
            case = _Case(name)
            self.assertFalse(case.settings.use_test_rng)
            with self.subTest(case=name):
                state = case.state_before(1)
                for index in range(1, len(case.steps)):
                    reference = case.steps[index]
                    step = moldyn_step(
                        case.settings, state, reference["forces"], reference["etot"]
                    )
                    self._compare_step(case, index, step, None, exact=True)
                    state = step.state

    def test_run_with_the_generator_of_gfortran_is_reproduced(self) -> None:
        """No ``test_rng``: the Fortran run drew from gfortran's random_number."""
        case = _Case("nve_intrinsic_rng")
        self.assertFalse(case.settings.use_test_rng)
        self.assertEqual(case.settings.rng_seed, 12345)
        random = case.settings.random_source(compiler="gfortran")
        self.assertIsInstance(random, GfortranRandom)
        state = case.initial_state()
        records = initial_records(state, case.species, case.first_positions)
        for index, reference in enumerate(case.steps):
            step = moldyn_step(
                case.settings, state, reference["forces"], reference["etot"], random=random
            )
            self._compare_step(case, index, step, random, exact=False)
            records = records + step_records(state, step, case.species)
            state = step.state
        # 5 atoms, 3 velocities each, at least two uniform numbers per deviate
        self.assertGreater(random.position, 30)
        files = case.data["files"]
        self.assertEqual(records.md_nrg, files["md_nrg.dat"])
        self.assertEqual(records.md_mech, files["md_mech.dat"])
        self.assertEqual(records.atom_cor, files["atom.cor"])
        self.assertEqual(records.movie_xyz, files["movie.xyz"])
        with self.assertRaises(ValueError):
            case.settings.random_source(compiler="ifort")
        with self.assertRaises(ValueError):
            MolecularDynamicsSettings(enabled=True).random_source(compiler="gfortran")

    def test_forces_evaluated_in_python_reproduce_the_runs(self) -> None:
        """The whole loop of parsec.F90 with the Morse cluster evaluated here."""
        worst = 0.0
        for name in REPEATABLE:
            case = _Case(name)
            with self.subTest(case=name):
                run = run_molecular_dynamics(
                    case.settings,
                    case.initial_state(),
                    lambda positions: _morse(positions, case.species),
                    first_positions=case.first_positions,
                )
                self.assertEqual(run.evaluations, case.data["scf_count"])
                self.assertEqual(len(run.steps), len(case.steps))
                self.assertIsNone(run.records)
                for step, reference in zip(run.steps, case.steps):
                    theirs = np.array(reference["after"]["xcur"])
                    np.testing.assert_allclose(
                        step.positions, theirs, rtol=1.0e-11, atol=1.0e-300
                    )
                    np.testing.assert_allclose(
                        step.state.velocities,
                        reference["after"]["vold"],
                        rtol=1.0e-10,
                        atol=1.0e-300,
                    )
                    scale = np.where(theirs == 0.0, 1.0, np.abs(theirs))
                    worst = max(
                        worst, float(np.max(np.abs(step.positions - theirs) / scale))
                    )
        self.assertLess(worst, 1.0e-11)

    def test_number_of_calls_and_the_last_step(self) -> None:
        # the final temperature is reached: the last call is the "last step"
        case = _Case("nve_stair")
        state = case.initial_state()
        random = case.settings.random_source()
        flags = []
        for reference in case.steps:
            step = moldyn_step(
                case.settings, state, reference["forces"], reference["etot"], random=random
            )
            flags.append((step.last_step, step.finished))
            state = step.state
        self.assertEqual(flags, [(False, False)] * 6 + [(True, True)])
        # the stair did not get to T_Final: there is no "last step"
        case = _Case("nve_stair_unfinished")
        state = case.initial_state()
        random = case.settings.random_source()
        flags = []
        for reference in case.steps:
            step = moldyn_step(
                case.settings, state, reference["forces"], reference["etot"], random=random
            )
            flags.append((step.last_step, step.finished))
            state = step.state
        self.assertEqual(flags, [(False, False), (False, False), (False, True)])
        self.assertEqual(state.temperature, 300.0)

    def test_step_num_zero_makes_one_evaluation_and_no_step(self) -> None:
        case = _Case("zero_steps")
        self.assertEqual(case.data["moldyn_calls"], 0)
        self.assertEqual(case.data["scf_count"], 1)
        state = case.initial_state()
        run = run_molecular_dynamics(
            case.settings,
            state,
            lambda positions: _morse(positions, case.species),
            species=case.species,
        )
        self.assertEqual(run.evaluations, 1)
        self.assertEqual(run.steps, ())
        self.assertIs(run.state, state)
        with self.assertRaises(ValueError):
            moldyn_step(case.settings, state, np.zeros((5, 3)), 0.0)
        files = case.data["files"]
        self.assertEqual(run.records.md_nrg, files["md_nrg.dat"])
        self.assertEqual(run.records.md_mech, files["md_mech.dat"])
        self.assertEqual(run.records.atom_cor, files["atom.cor"])
        self.assertEqual(run.records.movie_xyz, files["movie.xyz"])

    def test_temperature_schedules(self) -> None:
        def schedule(name):
            case = _Case(name)
            targets = [step["energy"]["tempi"] for step in case.steps]
            return case, targets

        _, stair = schedule("nve_stair")
        # two steps per stair; the third stair is clamped to T_Final
        self.assertEqual(stair, [600.0, 600.0, 500.0, 500.0, 400.0, 400.0, 350.0])
        _, heating = schedule("langevin_stair_heating")
        self.assertEqual(
            heating, [300.0, 300.0, 400.0, 400.0, 500.0, 500.0, 600.0, 600.0, 600.0]
        )
        case, linear = schedule("nve_linear")
        np.testing.assert_allclose(linear, np.linspace(500.0, 200.0, 7), rtol=1e-14)
        case, log = schedule("nve_log")
        np.testing.assert_allclose(
            log, 800.0 * (100.0 / 800.0) ** (np.arange(7) / 6.0), rtol=1e-14
        )

    def test_fixed_atoms_and_centre_of_mass(self) -> None:
        case = _Case("nve_stair")
        fixed = np.flatnonzero(case.movable == 0)
        self.assertEqual(list(fixed), [2])
        state = case.initial_state()
        random = case.settings.random_source()
        for reference in case.steps:
            step = moldyn_step(
                case.settings, state, reference["forces"], reference["etot"], random=random
            )
            state = step.state
            self.assertTrue(
                np.array_equal(state.current_positions[fixed], case.first_positions[fixed])
            )
            self.assertTrue(np.all(state.velocities[fixed] == 0.0))
            self.assertTrue(np.all(state.accelerations[fixed] == 0.0))
        # Newtonian branch: the velocity at the current positions has no
        # centre-of-mass component; it is (xcur - xold)/dt + dt (2a + a_old)/6
        # and its kinetic energy is the one reported.
        dt = case.settings.time_step
        previous = case.state_before(3)
        reference = case.steps[3]
        step = moldyn_step(case.settings, previous, reference["forces"], reference["etot"])
        mobile = case.movable[:, None]
        acceleration = 0.5 * np.array(reference["forces"]) / case.masses[:, None] * mobile
        velocity = (
            previous.current_positions - previous.previous_positions
        ) / dt + dt * (2.0 * acceleration + previous.accelerations) / 6.0
        velocity = velocity * mobile
        momentum = (case.masses[:, None] * velocity).sum(axis=0)
        velocity = velocity - momentum / (case.masses * case.movable).sum() * mobile
        self.assertLess(
            np.max(np.abs((case.masses[:, None] * velocity).sum(axis=0))), 1.0e-12
        )
        kinetic_hartree = 0.5 * float((case.masses[:, None] * velocity**2).sum())
        self.assertAlmostEqual(step.kinetic_energy, 2.0 * kinetic_hartree, places=14)
        self.assertAlmostEqual(
            step.conserved_energy, reference["etot"] + 2.0 * kinetic_hartree, places=13
        )

    def test_reported_energies_and_temperatures(self) -> None:
        # first step: the velocities are scaled to the input temperature
        for name in ("nve_stair", "langevin_linear", "langevin_log"):
            case = _Case(name)
            step = moldyn_step(
                case.settings,
                case.initial_state(),
                case.steps[0]["forces"],
                case.steps[0]["etot"],
                random=case.settings.random_source(),
            )
            mobile = int(case.movable.sum())
            self.assertEqual(step.time_ps, 0.0)
            self.assertAlmostEqual(
                step.temperature / case.settings.initial_temperature, 1.0, places=12
            )
            # per movable atom in eV; the potential energy is per atom
            self.assertAlmostEqual(
                step.kinetic_energy_ev, step.kinetic_energy / mobile * 13.6058, places=14
            )
            self.assertEqual(
                step.potential_energy_ev, 13.6058 * case.steps[0]["etot"] / len(case.species)
            )
            self.assertEqual(
                step.total_energy_ev, step.kinetic_energy_ev + step.potential_energy_ev
            )
        # 1e-6 <= friction < 1e-5: Langevin integrator, yet the first
        # velocities are scaled with 3N degrees of freedom and the reported
        # temperature uses 3(N-1): it starts above the input temperature
        case = _Case("langevin_weak_friction")
        self.assertFalse(case.settings.is_microcanonical)
        self.assertAlmostEqual(
            _number(case.steps[0]["energy"]["tcalc"]), 400.0 * 4.0 / 3.0, places=9
        )
        # one movable atom: the Langevin temperature divides by N - 1 = 0
        case = _Case("langevin_single_mobile")
        self.assertEqual(int(case.movable.sum()), 1)
        step = moldyn_step(
            case.settings,
            case.initial_state(),
            case.steps[0]["forces"],
            case.steps[0]["etot"],
            random=case.settings.random_source(),
        )
        self.assertEqual(step.temperature, math.inf)
        self.assertEqual(_number(case.steps[0]["energy"]["tcalc"]), math.inf)

    def test_energy_is_conserved_to_second_order_in_the_time_step(self) -> None:
        """Beeman integration of the Morse cluster, forces evaluated here.

        All atoms move.  The cluster starts far from equilibrium, so the
        kinetic energy changes by several tenths of a rydberg; the sum of
        potential and kinetic energy must stay constant up to an error that
        falls by four when the time step is halved.
        """
        case = _Case("langevin_log")
        self.assertEqual(list(case.movable), [1] * 5)
        spread = {}
        for time_step, count in ((5.0, 200), (2.5, 400)):
            settings = MolecularDynamicsSettings(
                enabled=True,
                initial_temperature=300.0,
                final_temperature=300.0,
                time_step=time_step,
                step_num=count,
                use_test_rng=True,
                rng_seed=1,
            )
            state = start_molecular_dynamics(settings, case.first_positions, case.masses)
            run = run_molecular_dynamics(
                settings, state, lambda positions: _morse(positions, case.species)
            )
            self.assertEqual(len(run.steps), count + 1)
            conserved = np.array([step.conserved_energy for step in run.steps])
            kinetic = np.array([step.kinetic_energy for step in run.steps])
            self.assertGreater(np.ptp(kinetic), 0.3)
            spread[time_step] = float(np.ptp(conserved))
            # the centre of mass stays at rest
            for step in run.steps:
                self.assertLess(np.max(np.abs(step.center_of_mass_velocity)), 1.0e-15)
        self.assertLess(spread[5.0], 3.0e-3)
        self.assertLess(spread[2.5], 0.3 * spread[5.0])
        self.assertGreater(spread[2.5], 0.2 * spread[5.0])

    def test_langevin_bath_holds_free_atoms_at_the_target_temperature(self) -> None:
        """Friction and random force balance (no forces, PARSEC's own generator).

        ``moldyn`` reports the Langevin temperature with ``N/(N-1)``, so the
        long-time mean is the target times that factor, here within the
        statistical error of 400 steps of 30 atoms.
        """
        count = 30
        settings = MolecularDynamicsSettings(
            enabled=True,
            initial_temperature=600.0,
            final_temperature=600.0,
            friction_coefficient=0.01,
            time_step=20.0,
            step_num=600,
            use_test_rng=True,
            rng_seed=1,
        )
        state = start_molecular_dynamics(
            settings,
            np.arange(3.0 * count).reshape(count, 3),
            atomic_masses(["Si"] * 15 + ["H"] * 15),
        )
        run = run_molecular_dynamics(
            settings, state, lambda positions: (0.0, np.zeros((count, 3)))
        )
        temperatures = np.array([step.temperature for step in run.steps])
        self.assertAlmostEqual(temperatures[0], 600.0, places=9)
        expected = 600.0 * count / (count - 1)
        self.assertLess(abs(temperatures[200:].mean() - expected), 0.08 * expected)
        self.assertGreater(temperatures[200:].std(), 30.0)

    def test_runs_with_a_numpy_generator_are_repeatable(self) -> None:
        """Without ``test_rng`` the numbers come from the generator given."""
        case = _Case("langevin_linear")
        settings = replace(case.settings, use_test_rng=False, rng_seed=0)

        def final_positions(seed):
            run = run_molecular_dynamics(
                settings,
                start_molecular_dynamics(
                    settings, case.first_positions, case.masses, case.movable
                ),
                lambda positions: _morse(positions, case.species),
                random=settings.random_source(seed),
            )
            self.assertEqual(len(run.steps), settings.step_num + 1)
            return run.state.current_positions

        first = final_positions(2026)
        self.assertTrue(np.array_equal(first, final_positions(2026)))
        self.assertTrue(
            np.array_equal(first, final_positions(np.random.default_rng(2026)))
        )
        self.assertFalse(np.array_equal(first, final_positions(2027)))
        self.assertTrue(np.all(np.isfinite(first)))

    def test_arguments_are_checked_and_states_are_not_modified(self) -> None:
        case = _Case("langevin_linear")
        state = case.initial_state()
        reference = case.steps[0]
        with self.assertRaises(ValueError):
            moldyn_step(case.settings, state, reference["forces"], reference["etot"])
        with self.assertRaises(ValueError):
            moldyn_step(
                case.settings,
                state,
                np.zeros((2, 3)),
                0.0,
                random=case.settings.random_source(),
            )
        before = state.current_positions.copy()
        step = moldyn_step(
            case.settings,
            state,
            reference["forces"],
            reference["etot"],
            random=case.settings.random_source(),
        )
        self.assertTrue(np.array_equal(state.current_positions, before))
        self.assertEqual(state.frame, 0)
        self.assertTrue(state.is_first_step)
        self.assertFalse(step.state.is_first_step)
        with self.assertRaises(ValueError):
            step.state.velocities[0, 0] = 1.0
        with self.assertRaises(ValueError):
            start_molecular_dynamics(
                case.settings, case.first_positions, case.masses, np.zeros(5, dtype=int)
            )
        with self.assertRaises(ValueError):
            start_molecular_dynamics(case.settings, case.first_positions[:, :2], case.masses)
        everyone = start_molecular_dynamics(
            case.settings, case.first_positions, case.masses
        )
        self.assertEqual(list(everyone.movable), [1] * 5)


class FileTests(unittest.TestCase):
    def test_files_are_those_of_the_fortran_runs(self) -> None:
        """md_nrg.dat, md_mech.dat, atom.cor and movie.xyz, character for character."""
        compared = 0
        for name in REPEATABLE:
            case = _Case(name)
            with self.subTest(case=name):
                state = case.initial_state()
                random = case.settings.random_source()
                records = initial_records(state, case.species, case.first_positions)
                for reference in case.steps:
                    step = moldyn_step(
                        case.settings, state, reference["forces"], reference["etot"],
                        random=random,
                    )
                    records = records + step_records(state, step, case.species)
                    state = step.state
                files = case.data["files"]
                self.assertEqual(records.md_nrg, files["md_nrg.dat"])
                self.assertEqual(records.md_mech, files["md_mech.dat"])
                self.assertEqual(records.atom_cor, files["atom.cor"])
                self.assertEqual(records.movie_xyz, files["movie.xyz"])
                compared += 4
        self.assertGreaterEqual(compared, 60)

    def test_parsec_out_paragraph_is_that_of_usrinput(self) -> None:
        compared = 0
        for name, case in CASES.items():
            if "parsec_out" not in case or case["usrinput_ierr"]:
                continue
            with self.subTest(case=name):
                keywords = case["keywords"]
                try:
                    settings = MolecularDynamicsSettings.from_keywords(keywords)
                except MolecularDynamicsInputError:
                    # usrinput prints the paragraph; initial then stops
                    self.assertIn(case["initial_ierr"], (231, 232))
                    continue
                self.assertEqual(format_parsec_out_summary(settings), case["parsec_out"])
                compared += 1
        self.assertGreaterEqual(compared, 18)
        self.assertEqual(
            format_parsec_out_summary(MolecularDynamicsSettings()),
            " No molecular dynamics!\n",
        )

    def test_headers(self) -> None:
        files = CASES["nve_stair"]["files"]
        self.assertTrue(files["md_nrg.dat"].startswith(MD_NRG_HEADER))
        self.assertTrue(files["md_mech.dat"].startswith(MD_MECH_HEADER))
        self.assertEqual(MD_NRG_HEADER.count("\n"), 1)
        self.assertEqual(MD_MECH_HEADER.count("\n"), 5)

    def test_run_helper_collects_the_same_files(self) -> None:
        case = _Case("langevin_log")
        run = run_molecular_dynamics(
            case.settings,
            case.initial_state(),
            lambda positions: _morse(positions, case.species),
            species=case.species,
        )
        files = case.data["files"]
        # eight or nine printed digits: last-bit differences of exp() in the
        # forces do not show
        self.assertEqual(run.records.md_nrg, files["md_nrg.dat"])
        self.assertEqual(run.records.atom_cor, files["atom.cor"])
        self.assertEqual(run.records.movie_xyz, files["movie.xyz"])

    def test_silver_benchmark_layout(self) -> None:
        """ref/md_nrg.dat of PARSEC's 0d_silver_cluster_md example.

        The energies of that run need its electronic structure.  Step, time,
        target temperature and the layout of every line do not: the Python
        loop is run with zero forces for the schedule, the energies are
        taken from the reference line, and the line must come out again,
        asterisks included.  The first line is more than layout: the initial
        velocities are scaled to the input temperature whatever the random
        numbers are, so its kinetic energy and its temperature are the
        Python values.
        """
        silver = REFERENCE["silver_cluster_md_nrg"]
        lines = silver["md_nrg_dat"].splitlines()
        self.assertEqual(lines[0] + "\n", MD_NRG_HEADER)
        settings = MolecularDynamicsSettings.from_keywords(silver["keywords"])
        self.assertEqual(settings.cooling_method, "log")
        self.assertEqual(settings.time_step, 150.0)
        self.assertTrue(settings.is_microcanonical)
        count = silver["atom_count"]
        positions = np.arange(3.0 * count).reshape(count, 3)
        state = start_molecular_dynamics(
            settings, positions, atomic_masses([silver["species"]] * count)
        )
        run = run_molecular_dynamics(
            settings,
            state,
            lambda positions: (0.0, np.zeros((count, 3))),
            random=ParsecTestRandom(1),
        )
        self.assertEqual(len(run.steps), len(lines) - 1)
        self.assertEqual(len(run.steps), settings.step_num + 1)
        for index, (step, line) in enumerate(zip(run.steps, lines[1:])):
            kinetic = float(line[16:27])
            potential = float(line[29:40])
            total = float(line[42:53])
            if index == 0:
                kinetic = step.kinetic_energy_ev
                temperature = step.temperature
                self.assertAlmostEqual(temperature, 2000.0, places=9)
            elif "*" in line[65:72]:
                # 2/3 Ekin/k of the microcanonical branch does not fit f7.2
                temperature = 2.0 / 3.0 * (kinetic / 13.6058) / 6.33327186e-06
                self.assertGreater(temperature, 9999.995)
            else:
                temperature = float(line[65:72])
            rebuilt = replace(
                step,
                kinetic_energy_ev=kinetic,
                potential_energy_ev=potential,
                total_energy_ev=total,
                temperature=temperature,
            )
            self.assertEqual(format_md_nrg_line(rebuilt), line + "\n")
            self.assertEqual(len(line), 72)

    def test_single_blocks(self) -> None:
        state = MolecularDynamicsState(
            masses=[1850.0, 1850.0],
            movable=[1, 0],
            current_positions=[[1.0, -2.0, 0.75], [0.0, 0.0, 0.0]],
            previous_positions=[[1.0, -2.0, 0.25], [0.0, 0.0, 0.0]],
            velocities=[[0.0, 0.0, 1.25e-3], [0.0, 0.0, 0.0]],
            accelerations=[[0.0, 0.0, -1.0e-7], [0.0, 0.0, 0.0]],
            temperature=300.0,
            frame=12,
            move_count=12,
        )
        block = format_md_mech_step(state).splitlines()
        self.assertEqual(block[0], "Step #  12")
        self.assertEqual(len(block), 3)
        self.assertEqual(len(block[1]), 13 * 16)
        self.assertEqual(
            block[1][:48], "  0.10000000E+01 -0.20000000E+01  0.75000000E+00"
        )
        self.assertTrue(block[1].endswith(" -0.10000000E-06  0.10000000E+01"))
        self.assertTrue(block[2].endswith("  0.00000000E+00"))
        self.assertEqual(format_md_restart(state), format_md_mech_step(state))
        # from step 10000 on the header overflows its i4 field
        self.assertTrue(
            format_md_mech_step(replace(state, frame=10000)).startswith("Step #****\n")
        )
        cor = format_atom_cor_step(["Si", "H"], state.current_positions, 3)
        self.assertEqual(
            cor.splitlines(),
            [
                " Coordinates for step #           3",
                " Si",
                "     1.000000000    -2.000000000     0.750000000",
                " H ",
                "     0.000000000     0.000000000     0.000000000",
            ],
        )
        xyz = format_movie_xyz_frame(["Si", "H"], state.current_positions, 3)
        self.assertEqual(
            xyz.splitlines(),
            [
                "  2",
                " Step #            3",
                "Si     0.529177    -1.058354     0.396883",
                "H      0.000000     0.000000     0.000000",
            ],
        )
        # two atom types with the same name next to each other form one group
        grouped = format_atom_cor_step(["H", "H"], state.current_positions, 0)
        self.assertEqual(grouped.count(" H \n"), 1)
        # i3 for the number of atoms
        many = format_movie_xyz_frame(["H"] * 1000, np.zeros((1000, 3)), 0)
        self.assertTrue(many.startswith("***\n"))
        with self.assertRaises(ValueError):
            format_atom_cor_step(["Si"], state.current_positions, 3)


class RestartTests(unittest.TestCase):
    def test_restart_file_is_read_as_initial_reads_it(self) -> None:
        for name in ("restart_nve", "restart_langevin", "restart_shifted_cluster"):
            case = _Case(name)
            with self.subTest(case=name):
                fortran = case.data["state0"]
                state = case.initial_state()
                self.assertTrue(state.restarted)
                self.assertFalse(state.is_first_step)
                self.assertEqual(state.frame, case.data["settings"]["iframe"])
                self.assertEqual(state.frame, case.data["restart_from"]["frame"])
                self.assertEqual(state.move_count, 0)
                self.assertEqual(state.temperature, case.settings.initial_temperature)
                self.assertTrue(np.array_equal(state.current_positions, fortran["xcur"]))
                self.assertTrue(np.array_equal(state.previous_positions, fortran["xold"]))
                self.assertTrue(np.array_equal(state.velocities, fortran["vold"]))
                self.assertTrue(np.array_equal(state.accelerations, fortran["accold"]))
                # the flags of the file replace those of the input
                self.assertTrue(np.array_equal(state.movable, case.movable))

    def test_restart_file_written_here_is_the_fortran_one(self) -> None:
        """Run the parent case in Python and cut the restart file out of it."""
        for name in ("restart_nve", "restart_langevin", "restart_shifted_cluster"):
            with self.subTest(case=name):
                origin = CASES[name]["restart_from"]
                parent = _Case(origin["case"])
                state = parent.initial_state()
                random = parent.settings.random_source()
                md_mech = MD_MECH_HEADER
                for reference in parent.steps[: origin["frame"]]:
                    step = moldyn_step(
                        parent.settings, state, reference["forces"], reference["etot"],
                        random=random,
                    )
                    md_mech += format_md_mech_step(step.state)
                    state = step.state
                self.assertEqual(state.frame, origin["frame"])
                self.assertEqual(format_md_restart(state), CASES[name]["mdinit_dat"])
                self.assertEqual(last_md_mech_step(md_mech), CASES[name]["mdinit_dat"])

    def test_restart_continues_the_run_to_the_digits_of_the_file(self) -> None:
        """The restart file has eight digits: the continuation is that close."""
        restart = _Case("restart_nve")
        parent = _Case("nve_stair")
        frame = restart.data["restart_from"]["frame"]
        for index, reference in enumerate(restart.steps):
            uninterrupted = parent.steps[frame + index]["after"]
            np.testing.assert_allclose(
                reference["after"]["xcur"], uninterrupted["xcur"], rtol=0.0, atol=1.0e-6
            )
            self.assertFalse(
                np.array_equal(reference["after"]["xcur"], uninterrupted["xcur"])
            )
        # and Python continues exactly as Fortran continues
        state = restart.initial_state()
        for index, reference in enumerate(restart.steps):
            step = moldyn_step(
                restart.settings, state, reference["forces"], reference["etot"]
            )
            self.assertTrue(
                np.array_equal(step.positions, np.array(reference["after"]["xcur"]))
            )
            state = step.state

    def test_first_structure_of_a_restart_is_recentred_for_the_scf_only(self) -> None:
        """initial.f90 shifts clust%xatm but not the positions of the integrator."""
        from parsec_python import Atom, center_cluster_geometry

        # restart_nve: the cluster has drifted by more than 0.1 bohr since the
        # start of its parent run; the two others are still within 0.1 bohr
        for name, shifted in (("restart_nve", True), ("restart_shifted_cluster", False),
                              ("restart_langevin", False)):
            case = _Case(name)
            with self.subTest(case=name):
                state = case.initial_state()
                atoms = [
                    Atom(symbol, position)
                    for symbol, position in zip(case.species, state.current_positions)
                ]
                centred = np.array(
                    [atom.position for atom in center_cluster_geometry(atoms)]
                )
                # the structure of the first self-consistent calculation
                self.assertTrue(np.array_equal(centred, case.first_positions))
                self.assertTrue(
                    np.array_equal(centred, np.array(case.data["steps"][0]["positions"]))
                )
                self.assertEqual(
                    not np.array_equal(centred, state.current_positions), shifted
                )
                # the next structure continues from the unshifted positions
                reference = case.steps[0]
                step = moldyn_step(
                    case.settings, state, reference["forces"], reference["etot"],
                    random=case.settings.random_source(),
                )
                np.testing.assert_allclose(
                    step.positions, reference["after"]["xcur"], rtol=TRAJECTORY_RTOL
                )

    def test_fresh_start_uses_the_recentred_cluster(self) -> None:
        from parsec_python import Atom, center_cluster_geometry

        for name, shifted in (("nve_shifted_cluster", True), ("nve_stair", False)):
            case = _Case(name)
            given = case.data["input_atoms"]
            atoms = [
                Atom(symbol, position)
                for symbol, position in zip(given["species"], given["positions"])
            ]
            centred = np.array([atom.position for atom in center_cluster_geometry(atoms)])
            self.assertTrue(np.array_equal(centred, case.first_positions))
            self.assertTrue(np.array_equal(centred, np.array(case.data["state0"]["xcur"])))
            self.assertEqual(
                not np.array_equal(centred, np.array(given["positions"])), shifted
            )
            self.assertEqual(list(case.movable), given["movable"])

    def test_headers_and_short_lines(self) -> None:
        text = CASES["restart_nve"]["mdinit_dat"]
        restart = parse_md_restart(text)
        self.assertEqual(restart.frame, 3)
        self.assertEqual(restart.atom_count, 5)
        self.assertEqual(list(restart.movable), [1.0, 1.0, 0.0, 1.0, 1.0])
        # the header initial.f90 writes itself, with a list-directed integer
        lines = text.splitlines()
        relisted = "\n".join([" Step #           3"] + lines[1:]) + "\n"
        self.assertEqual(parse_md_restart(relisted).frame, 3)
        # a following step ends the block; an atom count stops earlier
        doubled = text + text
        self.assertEqual(parse_md_restart(doubled).atom_count, 5)
        self.assertEqual(parse_md_restart(text, 2).atom_count, 2)
        with self.assertRaises(ValueError):
            parse_md_restart(text, 6)
        # Fortran pads a short record with blanks: missing fields are zero
        short = "\n".join([lines[0]] + [line[:192] for line in lines[1:]]) + "\n"
        self.assertEqual(list(parse_md_restart(short).movable), [0.0] * 5)
        for bad in ("", "no header\n", "Step #****\n" + lines[1] + "\n", lines[0] + "\n"):
            with self.assertRaises(ValueError):
                parse_md_restart(bad)
        with self.assertRaises(ValueError):
            last_md_mech_step(MD_MECH_HEADER)

    def test_gfortran_cannot_read_the_restart_header(self) -> None:
        """Recorded behaviour of the distributed initial.f90, for the record."""
        probe = [line for line in REFERENCE["format_probe"] if line.startswith("RSTEP ")]
        self.assertNotEqual(int(probe[0].split()[1]), 0)
        for name in ("restart_nve", "restart_langevin"):
            failure = CASES[name]["distributed_initial_f90"]
            self.assertNotEqual(failure["return_code"], 0)
            self.assertIn("Constant string in input format", "\n".join(failure["stderr"]))


class ParsecExecutableTests(unittest.TestCase):
    """A run of the PARSEC program itself, not of the stand-alone driver.

    ``tests/data/moldyn_parsec_executable_reference.json`` holds the files
    PARSEC wrote in a short, complete microcanonical run of a water
    molecule.  The program prints its forces with six decimals only, but
    ``md_mech.dat`` lists the acceleration ``F/(2 m)`` of every step with
    eight significant digits.  The forces are taken from there; what Python
    computes from them, starting from the input file alone, is compared
    with the files: the initial velocities (PARSEC's own generator), the
    positions and velocities of every step, every line of ``md_nrg.dat``,
    and the last step with its "Coordinates for next Run".
    """

    @classmethod
    def setUpClass(cls) -> None:
        path = DATA.with_name("moldyn_parsec_executable_reference.json")
        cls.data = json.loads(path.read_text(encoding="utf-8"))

    def test_python_follows_the_executable(self) -> None:
        data = self.data
        species = data["species"]
        settings = MolecularDynamicsSettings.from_keywords(data["keywords"])
        self.assertTrue(settings.use_test_rng)
        self.assertTrue(settings.is_microcanonical)
        masses = atomic_masses(species)
        blocks = ["Step #" + part for part in data["files"]["md_mech.dat"].split("Step #")[1:]]
        self.assertEqual(len(blocks), data["steps"])
        # the run is complete: Step_num + 1 calls of moldyn
        self.assertTrue(data["complete"])
        self.assertEqual(len(blocks), settings.step_num + 1)
        state = start_molecular_dynamics(settings, data["positions"], masses)
        random = settings.random_source()
        records = initial_records(state, species)
        for index, block in enumerate(blocks):
            written = parse_md_restart(block, len(species))
            self.assertEqual(written.frame, index + 1)
            forces = 2.0 * masses[:, None] * written.accelerations
            # the same forces as parsec.out prints them
            np.testing.assert_allclose(
                forces, data["printed_forces"][index], rtol=0.0, atol=6.0e-7
            )
            step = moldyn_step(
                settings, state, forces, data["total_energies"][index], random=random
            )
            records = records + step_records(state, step, species)
            np.testing.assert_allclose(
                step.state.current_positions,
                written.current_positions,
                rtol=2.0e-7,
                atol=1.0e-9,
            )
            np.testing.assert_allclose(
                step.state.previous_positions,
                written.previous_positions,
                rtol=2.0e-7,
                atol=1.0e-9,
            )
            np.testing.assert_allclose(
                step.state.velocities, written.velocities, rtol=2.0e-6, atol=1.0e-10
            )
            state = step.state
            self.assertEqual(step.finished, index == len(blocks) - 1)
            self.assertEqual(step.last_step, index == len(blocks) - 1)
        files = data["files"]
        self.assertEqual(records.md_nrg, files["md_nrg.dat"])
        self.assertEqual(records.movie_xyz, files["movie.xyz"])
        self.assertIn("Coordinates for next Run", files["atom.cor"])
        # nine decimals in atom.cor: beyond the digits of the forces
        mine = np.array(
            [[float(x) for x in line.split()] for line in records.atom_cor.splitlines()
             if line.count(".") == 3]
        )
        theirs = np.array(
            [[float(x) for x in line.split()] for line in files["atom.cor"].splitlines()
             if line.count(".") == 3]
        )
        np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=2.0e-7)
        self.assertEqual(
            [line for line in records.atom_cor.splitlines() if line.count(".") != 3],
            [line for line in files["atom.cor"].splitlines() if line.count(".") != 3],
        )


if __name__ == "__main__":
    unittest.main()
