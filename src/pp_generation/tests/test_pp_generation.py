import os
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from pp_generation.backends.common import choose_local_channel
from pp_generation.errors import ConfigurationError, GhostStateError
from pp_generation.fhi_input import customize, parse
from pp_generation.generator import generate
from pp_generation.models import (
    BackendName,
    ConstructionScheme,
    CoreHole,
    GenerationRequest,
    GhostChannel,
    GhostStatus,
    LocalChannelResult,
    OutputFormat,
)
from pp_generation.parsers import atom_ghosts, fhi_ghosts

try:
    from parsec_python.Pseudopotential.potre import read_parsec_pseudopotential
except ImportError:  # package unit tests can run without the host DFT tree
    read_parsec_pseudopotential = None


# The Fortran PARSEC checkout with FHI98PP and ATOM comes from PARSEC_FORTRAN_ROOT; without it the tests that
# need it are skipped. The POTRE converter is this repository's own unless PARSEC_POTRE_CONVERTER names another.
PARSEC_ROOT = Path(os.environ["PARSEC_FORTRAN_ROOT"]) if os.environ.get("PARSEC_FORTRAN_ROOT") else None
FHI = PARSEC_ROOT / "fhi98pp" / "adka_v1_0" / "Dfhipp" if PARSEC_ROOT else None
ATOM_SRC = PARSEC_ROOT / "pseudopotential" / "Src" if PARSEC_ROOT else None
POTRE_CONVERTER = Path(
    os.environ.get("PARSEC_POTRE_CONVERTER", Path(__file__).resolve().parents[2] / "tools" / "upf_to_parsec.py")
)
NO_FHI = "FHI98PP reference checkout is unavailable (set PARSEC_FORTRAN_ROOT)"


@unittest.skipUnless(FHI is not None and FHI.is_dir(), NO_FHI)
class InputAndParserTests(unittest.TestCase):
    def test_core_hole_changes_ionic_charge(self) -> None:
        template = (FHI / "bin/Elements/TM/14-Si:tm.ini").read_text()
        modified = customize(
            template, xc="pbe", scheme=ConstructionScheme.TROULLIER_MARTINS,
            core_hole=CoreHole("2p", 1.0), cutoff_radius=2.5,
        )
        data = parse(modified)
        self.assertAlmostEqual(data.ionic_charge, 5.0)
        self.assertIn("2  1  5.00000000", modified)
        self.assertEqual(data.valence, ((3, 0, 2.0), (3, 1, 2.0)))

    def test_invalid_core_hole_is_rejected(self) -> None:
        template = (FHI / "bin/Elements/TM/14-Si:tm.ini").read_text()
        with self.assertRaises(ConfigurationError):
            customize(
                template, xc="pbe", scheme=ConstructionScheme.TROULLIER_MARTINS,
                core_hole=CoreHole("2p", 7.0), cutoff_radius=None,
            )

    def test_reference_fhi_ghost_report(self) -> None:
        report = FHI / "tests/si_2p_core_hole/generated/si_2p_ch_pbe_rc250/si_2p_ch_pbe_rc250.test"
        channels = fhi_ghosts(report)
        self.assertEqual([item.label for item in channels], ["s", "p"])
        self.assertTrue(all(item.status is GhostStatus.PASS for item in channels))

    def test_atom_criterion_detects_synthetic_ghost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "atom.out"
            report.write_text(
                "Ghost State Test (X. Gonze et al)\n\n"
                " l j 0-node-eigen 1-node-eigen True-eigen inorm\n\n"
                " 0 -1.000000 -0.100000 -0.050000 -1\n"
            )
            channels = atom_ghosts(report)
            self.assertEqual(channels[0].status, GhostStatus.GHOST)
            self.assertLess(channels[0].margin_hartree, 0)

    def test_local_selection_rejects_failed_requested_channel(self) -> None:
        failed = LocalChannelResult(
            0, (GhostChannel(1, "p", GhostStatus.GHOST, "ghost", -0.1),), Path("failed")
        )
        safe = LocalChannelResult(
            1, (GhostChannel(0, "s", GhostStatus.PASS, "pass", 0.2),), Path("safe")
        )
        self.assertEqual(choose_local_channel([failed, safe], requested=None, reject_ghosts=True), safe)
        with self.assertRaises(GhostStateError):
            choose_local_channel([failed, safe], requested=0, reject_ghosts=True)


@unittest.skipUnless(FHI is not None and FHI.is_dir(), NO_FHI)
class FhiIntegrationTests(unittest.TestCase):
    def test_si_2p_core_hole_is_byte_identical(self) -> None:
        reference = FHI / "tests/si_2p_core_hole/generated/si_2p_ch_pbe_rc250"
        with tempfile.TemporaryDirectory() as directory:
            result = generate(
                GenerationRequest(
                    "Si", Path(directory), fhi_root=FHI,
                    core_hole=CoreHole("2p"), cutoff_radius=2.5,
                    prefix="si_2p_ch_pbe_rc250", formats=(OutputFormat.FHI,),
                )
            )
            self.assertTrue(result.ghost_free)
            self.assertEqual(result.selected_local_channel, 2)
            self.assertAlmostEqual(result.ae_energy_hartree or 0, -285.79258, places=5)
            self.assertAlmostEqual(result.pseudo_energy_hartree or 0, -5.87811, places=5)
            self.assertEqual(result.artifacts["cpi"].read_bytes(), (reference / "si_2p_ch_pbe_rc250.cpi").read_bytes())

    def test_p_1s_all_formats_are_byte_identical(self) -> None:
        qe = FHI / "tools/fhi2upf_qe.x"
        converter = POTRE_CONVERTER
        if not qe.is_file() or not converter.is_file():
            self.skipTest("QE or POTRE converter is unavailable")
        reference = FHI / "tests/p_1s_core_hole"
        with tempfile.TemporaryDirectory() as directory:
            result = generate(
                GenerationRequest(
                    "P", Path(directory), fhi_root=FHI, core_hole=CoreHole("1s"),
                    cutoff_radius=1.95, prefix="p_1s_ch_pbe_rc195",
                    formats=(OutputFormat.FHI, OutputFormat.UPF, OutputFormat.PARSEC),
                    qe_converter=qe, potre_converter=converter,
                )
            )
            expected = {
                "fhi": reference / "converted/core_hole/P_FHIPP.DAT",
                "upf": reference / "converted/core_hole/p_1s_ch_pbe_rc195.UPF",
                "parsec": reference / "converted/core_hole/P_POTRE.DAT",
            }
            for name, path in expected.items():
                self.assertEqual(result.artifacts[name].read_bytes(), path.read_bytes(), name)
            self.assertEqual(
                result.artifacts["fhi_species"].name,
                "P-1s_FHIPP.DAT",
            )
            self.assertEqual(
                result.artifacts["parsec_species"].name,
                "P-1s_POTRE.DAT",
            )
            self.assertEqual(
                result.artifacts["fhi_species"].read_bytes(),
                result.artifacts["fhi"].read_bytes(),
            )
            self.assertEqual(
                result.artifacts["parsec_species"].read_bytes(),
                result.artifacts["parsec"].read_bytes(),
            )
            self.assertEqual(ET.parse(result.artifacts["upf"]).getroot().tag, "UPF")
            if read_parsec_pseudopotential is not None:
                parsed = read_parsec_pseudopotential(result.artifacts["parsec"])
                self.assertEqual(parsed.symbol, "P")
                self.assertAlmostEqual(parsed.ionic_charge, 6.0)
            report = json.loads(result.artifacts["report"].read_text())
            self.assertTrue(report["ghost_free"])
            self.assertEqual(len(report["artifacts"]["cpi"]["sha256"]), 64)

    def test_local_only_and_nlcc_potre_conversion(self) -> None:
        qe = FHI / "tools/fhi2upf_qe.x"
        converter = POTRE_CONVERTER
        if not qe.is_file() or not converter.is_file():
            self.skipTest("QE or POTRE converter is unavailable")
        if read_parsec_pseudopotential is None:
            self.skipTest("host PARSEC POTRE reader is unavailable")

        with tempfile.TemporaryDirectory() as directory:
            for element in ("He", "Li"):
                result = generate(
                    GenerationRequest(
                        element,
                        Path(directory) / element,
                        fhi_root=FHI,
                        formats=(OutputFormat.UPF, OutputFormat.PARSEC),
                        qe_converter=qe,
                        potre_converter=converter,
                    )
                )
                parsed = read_parsec_pseudopotential(result.artifacts["parsec"])
                self.assertEqual(parsed.symbol, element)
                self.assertEqual(
                    parsed.has_nonlinear_core_correction,
                    element == "Li",
                )
                if element == "He":
                    self.assertEqual(parsed.number_of_channels, 1)
                else:
                    self.assertGreater(float(parsed.core_density.max()), 0.0)


@unittest.skipUnless(
    ATOM_SRC is not None
    and (ATOM_SRC / "atom_all_gfortran.exe").is_file()
    and (ATOM_SRC / "kb_conv_gfortran.exe").is_file(),
    "ATOM executables are unavailable (set PARSEC_FORTRAN_ROOT)",
)
class AtomIntegrationTests(unittest.TestCase):
    def test_si_ca_all_formats_and_local_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = generate(
                GenerationRequest(
                    "Si", Path(directory), backend=BackendName.ATOM, xc="ca",
                    atom_executable=ATOM_SRC / "atom_all_gfortran.exe",
                    atom_kb_executable=ATOM_SRC / "kb_conv_gfortran.exe",
                    formats=(
                        OutputFormat.PARSEC, OutputFormat.UPF, OutputFormat.PSP8,
                        OutputFormat.SIESTA, OutputFormat.CPW2000,
                    ),
                )
            )
            self.assertTrue(result.ghost_free)
            self.assertEqual(result.selected_local_channel, 2)
            self.assertEqual({x.local_channel for x in result.local_channel_results}, {0, 1, 2})
            for output_format in result.request.formats:
                self.assertGreater(result.artifacts[output_format.value].stat().st_size, 100)
            self.assertEqual(ET.parse(result.artifacts["upf"]).getroot().tag, "UPF")
            if read_parsec_pseudopotential is not None:
                parsed = read_parsec_pseudopotential(result.artifacts["parsec"])
                self.assertEqual(parsed.symbol, "Si")
                self.assertAlmostEqual(parsed.ionic_charge, 4.0)


if __name__ == "__main__":
    unittest.main()
