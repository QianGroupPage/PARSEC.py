from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..errors import BackendError, ConfigurationError
from ..fhi_input import XC_CODES, ParsedInput, customize, find_template, parse
from ..formats import annotate_upf_energies
from ..models import (
    ConstructionScheme,
    GenerationRequest,
    GenerationResult,
    LocalChannelResult,
    OutputFormat,
    PseudopotentialFamily,
)
from ..parsers import energies, fhi_ghosts
from .common import choose_local_channel, run


SUPPORTED_FORMATS = {OutputFormat.FHI, OutputFormat.UPF, OutputFormat.PARSEC}


class Fhi98ppBackend:
    """Validated adapter for FHI98PP generation and transferability checks."""

    def _root(self, request: GenerationRequest) -> Path:
        value = request.fhi_root
        if value is None and "FHI98PP_ROOT" in os.environ:
            value = Path(os.environ["FHI98PP_ROOT"])
        if value is None and "FHIPP_ROOT" in os.environ:
            # Keep the historical spelling as a compatibility alias.
            value = Path(os.environ["FHIPP_ROOT"])
        if value is None:
            raise ConfigurationError(
                "set --fhi-root or FHI98PP_ROOT to the Dfhipp directory"
            )
        root = value.expanduser().resolve()
        for relative in ("bin/psgen", "bin/pswatch"):
            if not (root / relative).is_file():
                raise ConfigurationError(f"{root} is missing {relative}")
        return root

    def _validate(self, request: GenerationRequest) -> None:
        if request.family is not PseudopotentialFamily.NCPP:
            raise ConfigurationError("FHI98PP currently supports only the NCPP family")
        unknown = set(request.formats) - SUPPORTED_FORMATS
        if unknown:
            names = ", ".join(sorted(x.value for x in unknown))
            raise ConfigurationError(f"FHI98PP backend cannot write: {names}")
        if request.local_channel is not None and request.local_channel < 0:
            raise ConfigurationError("FHI98PP local channel must be non-negative")

    def _source(self, request: GenerationRequest, root: Path) -> str:
        if request.input_file:
            if request.core_hole is not None or request.cutoff_radius is not None:
                raise ConfigurationError(
                    "--input-file is authoritative; do not combine it with --core-hole or --cutoff-radius"
                )
            source = request.input_file.expanduser().read_text(encoding="utf-8")
            parsed = parse(source)
            # PBE and CA have user-friendly names in the template workflow.
            # An expert native input may use another FHI98PP XC code; in that
            # case ``request.xc`` is recorded as ``fhi-code-N`` and the input
            # remains fully authoritative.  UPF conversion is intentionally
            # restricted below because fhi2upf requires a recognized XC label.
            expected_xc = XC_CODES.get(request.xc)
            expected_scheme = (
                "t" if request.scheme is ConstructionScheme.TROULLIER_MARTINS else "h"
            )
            if (
                (expected_xc is not None and parsed.xc_code != expected_xc)
                or parsed.default_scheme != expected_scheme
            ):
                raise ConfigurationError(
                    "--xc/--scheme disagree with the authoritative FHI98PP --input-file"
                )
            return source
        template = find_template(root, request.element, request.scheme)
        return customize(
            template.read_text(encoding="utf-8"), xc=request.xc,
            scheme=request.scheme, core_hole=request.core_hole,
            cutoff_radius=request.cutoff_radius,
        )

    def _test_candidates(
        self, request: GenerationRequest, root: Path, out: Path, prefix: str,
        input_path: Path, parsed: ParsedInput,
    ) -> tuple[list[LocalChannelResult], LocalChannelResult]:
        if request.local_channel is not None and request.local_channel > parsed.lmax:
            raise ConfigurationError(
                f"local channel l={request.local_channel} exceeds generated lmax={parsed.lmax}"
            )
        if request.local_channel is not None:
            local_channels = [request.local_channel]
        elif request.scan_local_channels:
            local_channels = list(range(parsed.lmax + 1))
        else:
            local_channels = [parsed.lmax]

        candidates: list[LocalChannelResult] = []
        for local in local_channels:
            run(
                [str(root / "bin/pswatch"), "-xv", "-q", "-l", str(local),
                 "-i", prefix, input_path.name],
                cwd=out,
            )
            generated = out / f"{prefix}.test"
            report = out / f"{prefix}.local-{local}.test"
            if not generated.is_file():
                raise BackendError(f"FHI98PP did not create {generated.name}")
            shutil.copy2(generated, report)
            candidates.append(LocalChannelResult(local, tuple(fhi_ghosts(report)), report))
        selected = choose_local_channel(
            candidates, requested=request.local_channel, reject_ghosts=request.reject_ghosts
        )
        if selected.local_channel != local_channels[-1]:
            run(
                [str(root / "bin/pswatch"), "-xv", "-q", "-l",
                 str(selected.local_channel), "-i", prefix, input_path.name],
                cwd=out,
            )
        return candidates, selected

    def _convert_upf(
        self, request: GenerationRequest, parsed: ParsedInput, source: str,
        result: GenerationResult, out: Path,
    ) -> Path:
        if request.xc not in {"pbe", "ca"}:
            raise ConfigurationError(
                "UPF/PARSEC conversion of an expert FHI98PP input requires a "
                "recognized --xc value (pbe or ca); native FHI output remains available"
            )
        converter = request.qe_converter
        if converter is None or not converter.expanduser().is_file():
            raise ConfigurationError(
                "UPF/PARSEC output from FHI98PP requires --qe-converter "
                "pointing to fhi2upf_qe.x"
            )
        lmax = parsed.lmax
        by_l = {ell: (n, occupation) for n, ell, occupation in parsed.valence}
        default_n = max(n for n, _, _ in parsed.valence)
        prompts = [request.element, request.xc.upper(), f"{lmax} {result.selected_local_channel}"]
        prompts.extend(
            f"{by_l.get(ell, (default_n, 0.0))[0]}{'spdfgh'[ell]} "
            f"{by_l.get(ell, (default_n, 0.0))[1]:g}"
            for ell in range(lmax + 1)
        )
        raw_upf = out / f"{result.prefix}.raw.UPF"
        run(
            [str(converter.expanduser().resolve()), str(result.artifacts["cpi"]), str(raw_upf)],
            cwd=out, stdin="\n".join(prompts) + "\n",
        )
        upf = out / f"{result.prefix}.UPF"
        annotate_upf_energies(result.artifacts["dat"], raw_upf, upf)
        result.artifacts["raw_upf"] = raw_upf
        return upf

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._validate(request)
        root = self._root(request)
        out = request.output_dir.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        prefix = request.prefix or _default_prefix(request)
        source = self._source(request, root)
        parsed = parse(source)
        input_path = out / f"{prefix}.ini"
        input_path.write_text(source, encoding="utf-8")

        run([str(root / "bin/psgen"), "-xv", "-o", prefix, input_path.name], cwd=out)
        result = GenerationResult(
            request=request, prefix=prefix, input_path=input_path,
            ionic_charge=parsed.ionic_charge, reference_electrons=parsed.reference_electrons,
        )
        result.warnings.append(
            "FHI98PP element templates are starting points; validate transferability and cutoff convergence."
        )
        reference_charge = parsed.ionic_charge - parsed.reference_electrons
        if abs(reference_charge) > 1.0e-10:
            result.warnings.append(
                f"Generating atomic reference charge {reference_charge:+g}; choose FCH/XCH electron "
                "count explicitly in the many-electron calculation."
            )
        for extension in ("dat", "cpi", "aep", "fc"):
            path = out / f"{prefix}.{extension}"
            if not path.is_file():
                raise BackendError(f"FHI98PP did not create required artifact {path.name}")
            result.artifacts[extension] = path
        if "not converged" in result.artifacts["dat"].read_text(errors="replace").lower():
            raise BackendError("FHI98PP output reports a non-converged atomic calculation")
        result.ae_energy_hartree, result.pseudo_energy_hartree = energies(result.artifacts["dat"])
        candidates, selected = self._test_candidates(
            request, root, out, prefix, input_path, parsed
        )
        result.local_channel_results = candidates
        result.selected_local_channel = selected.local_channel
        result.artifacts["test"] = selected.report_path
        for extension in (
            "lder", "density", "ps_ae_wfct", "ae_wfct", "pspot_s",
            "pspot_i", "unscreen",
        ):
            path = out / f"{prefix}.{extension}"
            if path.is_file():
                result.artifacts[extension] = path

        if OutputFormat.FHI in request.formats:
            destination = out / f"{request.element}_FHIPP.DAT"
            shutil.copy2(result.artifacts["cpi"], destination)
            result.artifacts[OutputFormat.FHI.value] = destination
            if request.core_hole is not None:
                species_fhi = out / f"{_core_hole_species_label(request)}_FHIPP.DAT"
                shutil.copy2(result.artifacts["cpi"], species_fhi)
                result.artifacts["fhi_species"] = species_fhi
        if set(request.formats) & {OutputFormat.UPF, OutputFormat.PARSEC}:
            result.artifacts[OutputFormat.UPF.value] = self._convert_upf(
                request, parsed, source, result, out
            )
        if OutputFormat.PARSEC in request.formats:
            converter = request.potre_converter
            if converter is None or not converter.expanduser().is_file():
                raise ConfigurationError(
                    "PARSEC POTRE output requires --potre-converter pointing to upf_to_parsec.py"
                )
            potre = out / f"{request.element}_POTRE.DAT"
            command = [os.environ.get("PYTHON", "python3"), str(converter.expanduser().resolve()), "--force"]
            if abs(parsed.reference_electrons - parsed.ionic_charge) > 1e-8:
                command.append("--allow-ionized-reference")
            command.extend([str(result.artifacts[OutputFormat.UPF.value]), str(potre)])
            run(command, cwd=out)
            result.artifacts[OutputFormat.PARSEC.value] = potre
            if request.core_hole is not None:
                # Preserve the historical element-only artifact for existing
                # PARSEC/FHI workflows, but also emit the configuration label
                # consumed directly by a Python Atom_Type such as C-1s.
                species_potre = (
                    out / f"{_core_hole_species_label(request)}_POTRE.DAT"
                )
                shutil.copy2(potre, species_potre)
                result.artifacts["parsec_species"] = species_potre
        return result


def _default_prefix(request: GenerationRequest) -> str:
    parts = [request.element.lower()]
    if request.core_hole:
        parts.append(f"{request.core_hole.shell.lower()}_hole_{request.core_hole.removed_electrons:g}")
    parts.extend((request.xc, request.scheme.value))
    return "_".join(parts)


def _core_hole_species_label(request: GenerationRequest) -> str:
    """Return the concise input species label for one core-hole reference."""

    if request.core_hole is None:
        return request.element
    charge = float(request.core_hole.removed_electrons)
    charge_suffix = "" if abs(charge - 1.0) <= 1.0e-12 else f"-{charge:g}"
    return f"{request.element}-{request.core_hole.shell.lower()}{charge_suffix}"
