"""A ctypes binding of the C library libxc (host arrays and CUDA device pointers).

libxc (S. Lehtola, C. Steigemann, M. J. T. Oliveira, M. A. L. Marques,
SoftwareX 7, 1 (2018); https://libxc.gitlab.io, MPL-2.0) is an optional,
dynamically loaded dependency of this package: nothing of its source is
part of the repository.  This module loads the shared library, declares the
public functions it calls and evaluates initialized functionals on

* host arrays (NumPy, C-contiguous float64), and
* device arrays (raw pointers of CuPy arrays) with a CUDA build of
  libxc 7.1 or later, whose ``xc_func_init_flags`` initializes a functional
  for the host (``XC_FLAGS_ON_HOST``) or for the device
  (``XC_FLAGS_ON_DEVICE``): one CUDA build serves both.

Variables and units are libxc's (Hartree atomic units): ``rho`` (spin
interleaved ``(np, 2)`` when polarized), ``sigma`` (``(np, 3)``:
``grad n_u . grad n_u``, ``grad n_u . grad n_d``, ``grad n_d . grad n_d``),
``lapl`` (``(np, 2)``), ``tau`` (``(np, 2)``, ``1/2 sum_i f_i |grad psi_i|**2``),
the energy per particle ``zk`` and the partial derivatives of the energy per
volume ``vrho``, ``vsigma``, ``vlapl``, ``vtau``.  The conversion to the
variables of this code lives in :mod:`parsec_python.V_xc.libxc`.

Why ctypes: no compiler is needed at installation (cffi's API mode needs
one, its ABI mode is ctypes again), the same code loads a libxc of
conda-forge, pip or a CMake build on Linux, Windows and macOS, and a device
pointer is a plain integer.  Only libxc's public accessor functions are
called; of ``xc_func_type`` only its stable leading members (``info``,
``nspin``, ``n_func_aux``, ``func_aux``, unchanged since libxc 5) are read,
to reach the components of a mixed functional.

The library is found through ``PARSEC_LIBXC_LIBRARY`` (the path of
``libxc.so``/``xc.dll``), else ``ctypes.util.find_library("xc")``, else the
usual file names on the loader's path.  ``scripts/build_libxc.sh`` builds
one (host or CUDA).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


#: ``nspin`` of ``xc_func_init``.
XC_UNPOLARIZED = 1
XC_POLARIZED = 2
#: ``xc_func_info_get_kind``.
XC_KINDS = {0: "exchange", 1: "correlation", 2: "exchange-correlation", 3: "kinetic"}
XC_KINETIC = 3
#: ``xc_func_info_get_family`` (``xc.h``).
XC_FAMILY_LDA = 1
XC_FAMILY_GGA = 2
XC_FAMILY_MGGA = 4
XC_FAMILY_LCA = 8
XC_FAMILY_OEP = 16
XC_FAMILY_HYB_GGA = 32
XC_FAMILY_HYB_MGGA = 64
XC_FAMILY_HYB_LDA = 128
#: Family code -> (family of the semilocal part, hybrid).
FAMILIES = {
    XC_FAMILY_LDA: ("lda", False),
    XC_FAMILY_GGA: ("gga", False),
    XC_FAMILY_MGGA: ("mgga", False),
    XC_FAMILY_HYB_LDA: ("lda", True),
    XC_FAMILY_HYB_GGA: ("gga", True),
    XC_FAMILY_HYB_MGGA: ("mgga", True),
    XC_FAMILY_LCA: ("lca", False),
    XC_FAMILY_OEP: ("oep", False),
}
#: ``xc_func_info_get_flags`` bits (libxc 7).
XC_FLAGS_HAVE_EXC = 1 << 0
XC_FLAGS_HAVE_VXC = 1 << 1
XC_FLAGS_1D = 1 << 5
XC_FLAGS_2D = 1 << 6
XC_FLAGS_3D = 1 << 7
XC_FLAGS_HYB_CAM = 1 << 8
XC_FLAGS_HYB_CAMY = 1 << 9
XC_FLAGS_VV10 = 1 << 10
XC_FLAGS_HYB_LC = 1 << 11
XC_FLAGS_HYB_LCY = 1 << 12
XC_FLAGS_STABLE = 1 << 13
XC_FLAGS_DEVELOPMENT = 1 << 14
XC_FLAGS_NEEDS_LAPLACIAN = 1 << 15
XC_FLAGS_NEEDS_TAU = 1 << 16
XC_FLAGS_ENFORCE_FHC = 1 << 17
XC_FLAGS_ON_DEVICE = 1 << 18
XC_FLAGS_ON_HOST = 1 << 19
#: The oldest libxc this binding accepts (the flags above, ``xc_func_set_fhc_enforcement``).
MINIMUM_VERSION = (7, 0, 0)
#: The oldest libxc whose CUDA build evaluates device pointers here (``xc_func_init_flags``).
MINIMUM_DEVICE_VERSION = (7, 1, 0)
#: The environment variable that names the library file.
LIBRARY_VARIABLE = "PARSEC_LIBXC_LIBRARY"

INSTALL_HINT = (
    "libxc (>= 7.0, https://libxc.gitlab.io) is an optional dependency: build it with "
    "scripts/build_libxc.sh (add --cuda for GPU runs) or install conda-forge's libxc-c, and set "
    f"{LIBRARY_VARIABLE} to the path of libxc.so (xc.dll on Windows)"
)

_double_p = ctypes.POINTER(ctypes.c_double)


class LibxcUnavailable(RuntimeError):
    """No usable libxc: not found, too old, or without the CUDA support a device evaluation needs."""


class _FuncPrefix(ctypes.Structure):
    """The leading members of libxc's ``xc_func_type`` (stable since libxc 5)."""


_FuncPrefix._fields_ = [
    ("info", ctypes.c_void_p),
    ("nspin", ctypes.c_int),
    ("n_func_aux", ctypes.c_int),
    ("func_aux", ctypes.POINTER(ctypes.POINTER(_FuncPrefix))),
]


def _candidates() -> list[str]:
    named = os.environ.get(LIBRARY_VARIABLE, "").strip()
    if named:
        return [named]
    found = ctypes.util.find_library("xc")
    names = [found] if found else []
    if sys.platform.startswith("win"):
        names += ["xc.dll", "libxc.dll"]
    elif sys.platform == "darwin":
        names += ["libxc.dylib"]
    else:
        names += ["libxc.so", "libxc.so.15", "libxc.so.12"]
    return names


def _declare(lib) -> None:
    c_int, c_double, c_char_p, c_void_p, c_size_t = (
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    signatures = {
        "xc_version": (None, [ctypes.POINTER(c_int)] * 3),
        "xc_version_string": (c_char_p, []),
        "xc_reference": (c_char_p, []),
        "xc_reference_doi": (c_char_p, []),
        "xc_functional_get_number": (c_int, [c_char_p]),
        "xc_functional_get_name": (c_void_p, [c_int]),
        "xc_family_from_id": (c_int, [c_int, ctypes.POINTER(c_int), ctypes.POINTER(c_int)]),
        "xc_number_of_functionals": (c_int, []),
        "xc_available_functional_numbers": (None, [ctypes.POINTER(c_int)]),
        "xc_func_alloc": (c_void_p, []),
        "xc_func_init": (c_int, [c_void_p, c_int, c_int]),
        "xc_func_end": (None, [c_void_p]),
        "xc_func_free": (None, [c_void_p]),
        "xc_func_get_info": (c_void_p, [c_void_p]),
        "xc_func_info_get_number": (c_int, [c_void_p]),
        "xc_func_info_get_kind": (c_int, [c_void_p]),
        "xc_func_info_get_name": (c_char_p, [c_void_p]),
        "xc_func_info_get_family": (c_int, [c_void_p]),
        "xc_func_info_get_flags": (c_int, [c_void_p]),
        "xc_func_info_get_references": (c_void_p, [c_void_p, c_int]),
        "xc_func_reference_get_ref": (c_char_p, [c_void_p]),
        "xc_func_reference_get_doi": (c_char_p, [c_void_p]),
        "xc_func_info_get_n_ext_params": (c_int, [c_void_p]),
        "xc_func_info_get_ext_params_name": (c_char_p, [c_void_p, c_int]),
        "xc_func_info_get_ext_params_description": (c_char_p, [c_void_p, c_int]),
        "xc_func_info_get_ext_params_default_value": (c_double, [c_void_p, c_int]),
        "xc_func_set_dens_threshold": (None, [c_void_p, c_double]),
        "xc_func_set_tau_threshold": (None, [c_void_p, c_double]),
        "xc_func_set_fhc_enforcement": (None, [c_void_p, c_int]),
        "xc_hyb_exx_coef": (c_double, [c_void_p]),
        "xc_hyb_cam_coef": (None, [c_void_p] + [ctypes.POINTER(c_double)] * 3),
        "xc_nlc_coef": (None, [c_void_p] + [ctypes.POINTER(c_double)] * 2),
        "xc_num_aux_funcs": (c_int, [c_void_p]),
        "xc_lda_exc_vxc": (None, [c_void_p, c_size_t] + [c_void_p] * 3),
        "xc_gga_exc_vxc": (None, [c_void_p, c_size_t] + [c_void_p] * 5),
        "xc_mgga_exc_vxc": (None, [c_void_p, c_size_t] + [c_void_p] * 9),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    for name, restype, argtypes in (
        ("xc_func_init_flags", c_int, [c_void_p, c_int, c_int, c_int]),
        ("xc_func_info_get_default_flags", c_int, []),
    ):
        function = getattr(lib, name, None)
        if function is not None:
            function.restype = restype
            function.argtypes = argtypes


@dataclass(frozen=True)
class FunctionalData:
    """What libxc says about one functional (no evaluation)."""

    number: int
    name: str
    description: str
    kind: str
    family_code: int
    flags: int
    alpha: float
    beta: float
    omega: float
    nlc_b: float
    nlc_c: float
    auxiliary_count: int
    references: tuple[str, ...]
    dois: tuple[str, ...]
    ext_params: tuple[tuple[str, float, str], ...]


class LibxcLibrary:
    """A loaded libxc: version, functional names and numbers, initialized functionals."""

    def __init__(self, path: str) -> None:
        try:
            lib = ctypes.CDLL(path)
        except OSError as error:
            raise LibxcUnavailable(f"libxc could not be loaded from {path!r} ({error}); {INSTALL_HINT}") from error
        try:
            _declare(lib)
        except AttributeError as error:
            raise LibxcUnavailable(f"{path!r} is not a libxc >= 7.0 ({error}); {INSTALL_HINT}") from error
        self._lib = lib
        self.path = getattr(lib, "_name", path)
        major, minor, micro = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        lib.xc_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(micro))
        self.version = (major.value, minor.value, micro.value)
        if self.version < MINIMUM_VERSION:
            raise LibxcUnavailable(
                f"libxc {self.version_string} at {self.path} is older than "
                f"{'.'.join(map(str, MINIMUM_VERSION))}; {INSTALL_HINT}"
            )
        self._init_flags = getattr(lib, "xc_func_init_flags", None)
        default = getattr(lib, "xc_func_info_get_default_flags", None)
        self.default_flags = int(default()) if default is not None else XC_FLAGS_ON_HOST
        #: A CUDA build whose functionals can be initialized for the device or the host.
        self.device_capable = bool(
            self._init_flags is not None and self.default_flags & XC_FLAGS_ON_DEVICE
        )
        # A CUDA build without xc_func_init_flags (libxc 7.0) evaluates device memory only.
        self.host_capable = self._init_flags is not None or not (self.default_flags & XC_FLAGS_ON_DEVICE)
        self._lock = threading.Lock()
        self._handles: dict[tuple, "LibxcHandle"] = {}
        self._data: dict[int, FunctionalData] = {}

    # -- names and numbers ---------------------------------------------------

    @property
    def version_string(self) -> str:
        return ".".join(str(value) for value in self.version)

    @property
    def reference(self) -> str:
        return self._lib.xc_reference().decode()

    @property
    def reference_doi(self) -> str:
        return self._lib.xc_reference_doi().decode()

    def number(self, name: str) -> int:
        """libxc's number of ``name`` (case-insensitive, with or without ``XC_``), -1 if unknown."""

        return int(self._lib.xc_functional_get_number(str(name).encode()))

    def name(self, number: int) -> str | None:
        """libxc's name of ``number`` (lowercase, ``gga_x_pbe``), ``None`` if unknown."""

        pointer = self._lib.xc_functional_get_name(int(number))
        if not pointer:
            return None
        text = ctypes.string_at(pointer).decode()
        # xc_functional_get_name returns a string the caller owns (malloc); a
        # few bytes per call are left to the process rather than freed through
        # another C runtime (Windows).
        return text

    def numbers(self) -> tuple[int, ...]:
        """All functional numbers of this libxc, ascending."""

        count = int(self._lib.xc_number_of_functionals())
        array = (ctypes.c_int * count)()
        self._lib.xc_available_functional_numbers(array)
        return tuple(sorted(int(value) for value in array))

    # -- functionals -----------------------------------------------------------

    def data(self, number: int) -> FunctionalData:
        """What libxc reports of functional ``number`` (an unpolarized host instance is initialized once)."""

        number = int(number)
        cached = self._data.get(number)
        if cached is not None:
            return cached
        handle = self.functional(number, polarized=False, device=None)
        lib = self._lib
        info = lib.xc_func_get_info(handle.pointer)
        references, dois = [], []
        for index in range(5):
            reference = lib.xc_func_info_get_references(info, index)
            if not reference:
                break
            references.append((lib.xc_func_reference_get_ref(reference) or b"").decode())
            dois.append((lib.xc_func_reference_get_doi(reference) or b"").decode())
        ext = []
        for index in range(int(lib.xc_func_info_get_n_ext_params(info))):
            ext.append(
                (
                    (lib.xc_func_info_get_ext_params_name(info, index) or b"").decode(),
                    float(lib.xc_func_info_get_ext_params_default_value(info, index)),
                    (lib.xc_func_info_get_ext_params_description(info, index) or b"").decode(),
                )
            )
        omega, alpha, beta = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        lib.xc_hyb_cam_coef(handle.pointer, ctypes.byref(omega), ctypes.byref(alpha), ctypes.byref(beta))
        nlc_b, nlc_c = ctypes.c_double(), ctypes.c_double()
        lib.xc_nlc_coef(handle.pointer, ctypes.byref(nlc_b), ctypes.byref(nlc_c))
        data = FunctionalData(
            number=number,
            name=self.name(number) or str(number),
            description=(lib.xc_func_info_get_name(info) or b"").decode(),
            kind=XC_KINDS.get(int(lib.xc_func_info_get_kind(info)), "unknown"),
            family_code=int(lib.xc_func_info_get_family(info)),
            flags=int(lib.xc_func_info_get_flags(info)),
            alpha=float(alpha.value),
            beta=float(beta.value),
            omega=float(omega.value),
            nlc_b=float(nlc_b.value),
            nlc_c=float(nlc_c.value),
            auxiliary_count=int(lib.xc_num_aux_funcs(handle.pointer)),
            references=tuple(references),
            dois=tuple(dois),
            ext_params=tuple(ext),
        )
        self._data[number] = data
        return data

    def functional(
        self,
        number: int,
        *,
        polarized: bool,
        device: int | None = None,
        enforce_fhc: bool | None = None,
        density_threshold: float | None = None,
        tau_threshold: float | None = None,
    ) -> "LibxcHandle":
        """The initialized functional ``number`` for host arrays (``device=None``) or device ``device``.

        One per (number, spin, device, FHC choice) and process.
        ``enforce_fhc`` sets libxc's Fermi-hole-curvature clamp of meta-GGAs
        (``sigma <= 8 rho tau``) on every component; ``density_threshold``
        replaces libxc's density threshold (``xc_func_set_dens_threshold``,
        every component), ``tau_threshold`` its floor of ``tau``
        (``xc_func_set_tau_threshold``, every component); ``None`` keeps
        libxc's default.
        """

        key = (
            int(number),
            bool(polarized),
            None if device is None else int(device),
            enforce_fhc,
            None if density_threshold is None else float(density_threshold),
            None if tau_threshold is None else float(tau_threshold),
        )
        handle = self._handles.get(key)
        if handle is not None:
            return handle
        with self._lock:
            handle = self._handles.get(key)
            if handle is None:
                handle = LibxcHandle(self, *key)
                self._handles[key] = handle
        return handle

    # -- internal --------------------------------------------------------------

    def _initialize(self, number: int, polarized: bool, device: int | None) -> int:
        lib = self._lib
        nspin = XC_POLARIZED if polarized else XC_UNPOLARIZED
        if device is not None:
            if not self.device_capable:
                raise LibxcUnavailable(
                    f"libxc {self.version_string} at {self.path} has no device support: a CUDA build of "
                    f"libxc >= {'.'.join(map(str, MINIMUM_DEVICE_VERSION))} evaluates on the GPU "
                    "(scripts/build_libxc.sh --cuda)"
                )
        elif not self.host_capable:
            raise LibxcUnavailable(
                f"libxc {self.version_string} at {self.path} is a CUDA build without "
                "xc_func_init_flags: it cannot evaluate host arrays (use libxc >= 7.1)"
            )
        pointer = lib.xc_func_alloc()
        if not pointer:
            raise MemoryError("xc_func_alloc failed")
        if self._init_flags is not None:
            flags = XC_FLAGS_ON_DEVICE if device is not None else XC_FLAGS_ON_HOST
            if device is not None:
                from ..acceleration.backends.cupy import require_cupy

                cp, _ = require_cupy()
                with cp.cuda.Device(int(device)):
                    status = self._init_flags(pointer, int(number), nspin, flags)
            else:
                status = self._init_flags(pointer, int(number), nspin, flags)
        else:
            status = lib.xc_func_init(pointer, int(number), nspin)
        if status != 0:
            lib.xc_func_free(pointer)
            raise ValueError(f"libxc has no functional number {number} (xc_func_init returned {status})")
        return int(pointer)

    def _set_fhc(self, pointer: int, on: bool) -> None:
        """``xc_func_set_fhc_enforcement`` on a functional and, recursively, on its components."""

        self._lib.xc_func_set_fhc_enforcement(pointer, 1 if on else 0)
        prefix = ctypes.cast(pointer, ctypes.POINTER(_FuncPrefix)).contents
        for index in range(int(prefix.n_func_aux)):
            component = prefix.func_aux[index]
            self._set_fhc(ctypes.cast(component, ctypes.c_void_p).value, on)


class LibxcHandle:
    """One initialized libxc functional (``xc_func_type``) for the host or one device.

    The evaluation methods take raw addresses (``ctypes`` integers) of
    C-contiguous float64 buffers of libxc's layout; :meth:`host_arrays` is
    the NumPy form.  Device buffers must be device or managed memory of
    ``device``; the call synchronizes the current CuPy stream before and the
    device after (libxc launches its kernel on the legacy default stream).
    """

    def __init__(
        self,
        library: LibxcLibrary,
        number: int,
        polarized: bool,
        device: int | None,
        enforce_fhc: bool | None,
        density_threshold: float | None = None,
        tau_threshold: float | None = None,
    ) -> None:
        self.library = library
        self.number = number
        self.polarized = polarized
        self.device = device
        self.pointer = library._initialize(number, polarized, device)
        if enforce_fhc is not None:
            library._set_fhc(self.pointer, bool(enforce_fhc))
        if density_threshold is not None:
            library._lib.xc_func_set_dens_threshold(self.pointer, float(density_threshold))
        if tau_threshold is not None:
            library._lib.xc_func_set_tau_threshold(self.pointer, float(tau_threshold))
        lib = library._lib
        info = lib.xc_func_get_info(self.pointer)
        self.family_code = int(lib.xc_func_info_get_family(info))
        self.flags = int(lib.xc_func_info_get_flags(info))
        self.family = FAMILIES.get(self.family_code, ("unknown", False))[0]

    @property
    def needs_tau(self) -> bool:
        return bool(self.flags & XC_FLAGS_NEEDS_TAU)

    @property
    def needs_laplacian(self) -> bool:
        return bool(self.flags & XC_FLAGS_NEEDS_LAPLACIAN)

    def _call(self, function, *arguments) -> None:
        if self.device is None:
            function(self.pointer, *arguments)
            return
        from ..acceleration.backends.cupy import require_cupy

        cp, _ = require_cupy()
        with cp.cuda.Device(self.device):
            cp.cuda.get_current_stream().synchronize()
            function(self.pointer, *arguments)
            cp.cuda.runtime.deviceSynchronize()

    def lda(self, count: int, rho: int, zk: int, vrho: int) -> None:
        self._call(self.library._lib.xc_lda_exc_vxc, int(count), rho, zk, vrho)

    def gga(self, count: int, rho: int, sigma: int, zk: int, vrho: int, vsigma: int) -> None:
        self._call(self.library._lib.xc_gga_exc_vxc, int(count), rho, sigma, zk, vrho, vsigma)

    def mgga(
        self,
        count: int,
        rho: int,
        sigma: int,
        lapl: int,
        tau: int,
        zk: int,
        vrho: int,
        vsigma: int,
        vlapl: int,
        vtau: int,
    ) -> None:
        self._call(
            self.library._lib.xc_mgga_exc_vxc,
            int(count),
            rho,
            sigma,
            lapl,
            tau,
            zk,
            vrho,
            vsigma,
            vlapl,
            vtau,
        )


_LIBRARY: LibxcLibrary | None = None
_LIBRARY_ERROR: str | None = None
_LOAD_LOCK = threading.Lock()


def load_libxc() -> LibxcLibrary:
    """The process's libxc (loaded once); :class:`LibxcUnavailable` with the install hint otherwise."""

    global _LIBRARY, _LIBRARY_ERROR
    if _LIBRARY is not None:
        return _LIBRARY
    with _LOAD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        errors = []
        for candidate in _candidates():
            if os.sep in candidate or "/" in candidate:
                if not Path(candidate).exists():
                    errors.append(f"{candidate}: no such file")
                    continue
            try:
                _LIBRARY = LibxcLibrary(candidate)
                _LIBRARY_ERROR = None
                return _LIBRARY
            except LibxcUnavailable as error:
                errors.append(str(error))
        _LIBRARY_ERROR = "; ".join(errors) if errors else "no candidate library name"
        raise LibxcUnavailable(f"no usable libxc ({_LIBRARY_ERROR}); {INSTALL_HINT}")


def libxc_available() -> bool:
    """Whether :func:`load_libxc` finds a usable library (no exception)."""

    try:
        load_libxc()
    except LibxcUnavailable:
        return False
    return True


def _reset_for_tests() -> None:
    """Forget the loaded library (tests that change ``PARSEC_LIBXC_LIBRARY``)."""

    global _LIBRARY, _LIBRARY_ERROR
    with _LOAD_LOCK:
        _LIBRARY = None
        _LIBRARY_ERROR = None


__all__ = [
    "FAMILIES",
    "FunctionalData",
    "INSTALL_HINT",
    "LIBRARY_VARIABLE",
    "LibxcHandle",
    "LibxcLibrary",
    "LibxcUnavailable",
    "MINIMUM_DEVICE_VERSION",
    "MINIMUM_VERSION",
    "XC_FLAGS_1D",
    "XC_FLAGS_2D",
    "XC_FLAGS_3D",
    "XC_FLAGS_DEVELOPMENT",
    "XC_FLAGS_ENFORCE_FHC",
    "XC_FLAGS_HAVE_EXC",
    "XC_FLAGS_HAVE_VXC",
    "XC_FLAGS_HYB_CAM",
    "XC_FLAGS_HYB_CAMY",
    "XC_FLAGS_HYB_LC",
    "XC_FLAGS_HYB_LCY",
    "XC_FLAGS_NEEDS_LAPLACIAN",
    "XC_FLAGS_NEEDS_TAU",
    "XC_FLAGS_ON_DEVICE",
    "XC_FLAGS_ON_HOST",
    "XC_FLAGS_VV10",
    "XC_KINETIC",
    "libxc_available",
    "load_libxc",
]
