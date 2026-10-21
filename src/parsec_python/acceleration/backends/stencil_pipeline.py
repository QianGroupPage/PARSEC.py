"""A software-pipelined slot loop for the recurrence kernel of the filter (``PARSEC_CUPY_STENCIL_KERNEL``).

``former`` compiles the kernels of ``cupy_stencil_major._CUDA_SOURCE`` and
``implicit_stencil.implicit_source`` exactly as before.  ``pipelined``, the default (also an unset
or empty variable), inserts in front of the slot loop of ``stencil_major_chebyshev6``, the fused
step of the Chebyshev filter, a path for the launches of production -- ``slot_count`` equal to the
slot count of the stencil the source was generated for, six columns, unit row stride -- in
straight-line code:

* the tile descriptor, the base and step of the slot records and the six column pointers are
  computed once per row (the former loop read the descriptor again in every slot);
* the slots are a software pipeline of their loads: the neighbour record and code of slot
  ``s + 2`` and the coefficient and six values of slot ``s + 1`` are loaded before the six FMAs
  of slot ``s``, so that a warp has two slots of value loads in flight where the former loop had
  at most one, and no slot waits behind a branch;
* a missing neighbour is skipped as before: its FMAs are not executed (they are predicated), and
  its loads read the row's own values (always in bounds) in place of a neighbour's;
* ``__launch_bounds__(256, 4)`` holds the kernel at 64 registers, four blocks of 256 threads per
  multiprocessor as the 56 registers of the former kernel;
* half-way through the slots, ``prefetch.global.L2`` hints for the first lines the epilogue reads
  (the potential, Y_{k-1}, the projector row offsets), which move no data into a register.

Per column the FMAs run in slot order in the expression of the former loop
(``accumulator[c] += coefficient * value``, an ``fma.rn.f64`` in the PTX of both); the epilogue
(scatter, potential, recurrence) is the former text, unchanged and after the same join, so ptxas
contracts it as before (its products ``center * x``, ``x * scale`` and ``sigma * y`` are fused by
ptxas, not by NVVM, and which one it fuses differs between the epilogue paths of the former
kernel: a copy of the epilogue inside the fast path changed it).  A launch of fewer columns,
another slot count or another row stride takes the former loop inside this kernel, with the bits
of the former kernel; the operator does not send it here, but to the former kernel, which it
compiles beside this one and which is faster outside the fast path
(``CuPyStencilMajorFiniteDifference.recurrence_kernel_for``).

``stencil_major_spmm6`` (``H X``) keeps the former text under every value and is the former
kernel itself.  Given the same fast path and launch bounds, ptxas commuted the operands of one of
its additions, ``accumulator[column] += nonlocal_accumulator[column]`` of the sixth column: the
sum is the same for every number, but where two NaNs of different payloads meet, the A100
returns the NaN of one operand position, so another NaN came out.

A generated kernel that cannot be built (its source or its compilation raises) gives way to the
former recurrence kernel: the operator keeps its layout and the former kernels, and so the former
bits, with a ``RuntimeWarning`` and the reason in its ``stencil_kernel_fallback``, instead of
failing over to another layout or to the CSR-order kernel.

What the stencils of a run launch is among its backend details, ``orbital_sector_stencil_kernel``
for the sectors of a symmetry eigensolver and ``gpu_stencil_kernel`` for a full-grid operator
(``CuPyStencilMajorFiniteDifference.stencil_kernel_record``): the value of the switch they were
built with, an unset variable included, the slot count of the fast path, or the fallback.
"""
from __future__ import annotations

import os
from typing import Any, NamedTuple
import warnings

SWITCH = "PARSEC_CUPY_STENCIL_KERNEL"
KERNELS = ("former", "pipelined")
# What an unset or empty PARSEC_CUPY_STENCIL_KERNEL gives; ``former`` restores the kernels of before.
DEFAULT = "pipelined"
THREADS = 256
BLOCKS_PER_MULTIPROCESSOR = 4
# The kernel that holds the fast path, and the columns of a launch that takes it.
KERNEL = "stencil_major_chebyshev6"
WIDTH = 6
_LOOP = "    for (int slot = 0; slot < slot_count; ++slot) {\n"
_VECTOR, _ROW_STRIDE, _COLUMN_STRIDE = "current", "current_row_stride", "current_column_stride"


def stencil_kernel_setting():
    """``PARSEC_CUPY_STENCIL_KERNEL``: ``former`` or ``pipelined``; unset or empty, :data:`DEFAULT`."""
    raw = os.environ.get(SWITCH, "").strip().lower()
    if not raw:
        return DEFAULT
    if raw in KERNELS:
        return raw
    raise ValueError(f"{SWITCH} must be former or pipelined")


def takes_fast_path(slot_count, width, row_stride, slots):
    """Whether a launch of the kernel generated for ``slots`` slots takes its fast path: its guard, on the host."""
    return int(slot_count) == int(slots) and int(width) == WIDTH and int(row_stride) == 1


def _fast_path(slots, tile):
    lines = []
    emit = lines.append

    def hint(address):
        emit(f'        asm volatile("prefetch.global.L2 [%0];" :: "l"({address}));')

    emit(f"    if (slot_count == {slots} && width == {WIDTH} && {_ROW_STRIDE} == 1) {{")
    if tile:
        emit(f"        const int descriptor = neighbors[row / {tile}];")
        emit("        const long long meta_base = descriptor < 0")
        emit("            ? static_cast<long long>(-descriptor)")
        emit(f"            : static_cast<long long>(descriptor) + row % {tile};")
        emit(f"        const int meta_step = descriptor < 0 ? 1 : {tile};")
        # A regular tile stores the offset of the neighbour; INT_MIN + row is negative for every row.
        emit("        const unsigned int row_shift = descriptor < 0 ? static_cast<unsigned int>(row) : 0u;")
    else:
        emit("        const long long meta_base = row;")
        emit("        const long long meta_step = row_count;")
        emit("        const unsigned int row_shift = 0u;")
    emit("        const int* __restrict__ meta_n = neighbors + meta_base;")
    emit("        const unsigned char* __restrict__ meta_c = coefficient_codes + meta_base;")
    for column in range(6):
        emit(f"        const double* __restrict__ column_{column} = {_VECTOR} + {column} * {_COLUMN_STRIDE};")
    emit("        const int self_row = static_cast<int>(row);")

    def at(slot):
        return f"{slot} * meta_step" if tile else f"static_cast<long long>({slot}) * meta_step"

    def records(slot):
        emit(f"        const int e{slot} = meta_n[{at(slot)}];")
        emit(f"        const unsigned int k{slot} = meta_c[{at(slot)}];")

    def values(slot):
        emit(f"        const int r{slot} = static_cast<int>(static_cast<unsigned int>(e{slot}) + row_shift);")
        emit(f"        const bool p{slot} = r{slot} >= 0;")
        emit(f"        const int q{slot} = p{slot} ? r{slot} : self_row;")
        emit(f"        const double w{slot} = coefficient_palette[p{slot} ? k{slot} : 0u];")
        for column in range(6):
            emit(f"        const double x{slot}_{column} = column_{column}[q{slot}];")

    def fmas(slot):
        emit(f"        if (p{slot}) {{")
        for column in range(6):
            emit(f"            accumulator[{column}] += w{slot} * x{slot}_{column};")
        emit("        }")

    def prefetches():
        hint("local_potential + row")
        emit("        if (add_previous) {")
        for column in range(6):
            hint(f"previous + row * previous_row_stride + static_cast<long long>({column}) * "
                 "previous_column_stride")
        emit("        }")
        emit("        if (add_nonlocal) {")
        hint("projector_row_offsets + row")
        emit("        }")

    # Step t: the records of slot t, the values of slot t - 1, the FMAs of slot t - 2.
    for step in range(slots + 2):
        if step < slots:
            records(step)
        if 0 <= step - 1 < slots:
            values(step - 1)
            if step - 1 == slots // 2:
                prefetches()
        if 0 <= step - 2 < slots:
            fmas(step - 2)
    emit("    } else")
    return "\n".join(lines) + "\n"


def pipelined_source(base, slots, tile):
    """``base`` -- the former source of the layout ``tile`` (0: slot-major) -- with the fast path in :data:`KERNEL`.

    The fast path goes in front of the slot loop of :data:`KERNEL`, whose former loop becomes its
    ``else`` branch, and the kernel is declared with launch bounds; every other byte of ``base``,
    ``stencil_major_spmm6`` among them, stays as it is.
    """
    if base.count(_LOOP) != 2:
        raise RuntimeError("production stencil ABI changed")
    if int(slots) < 1:
        raise ValueError("a stencil has at least one slot")
    start = base.index(f"void {KERNEL}(")
    at = base.index(_LOOP, start)
    source = base[:at] + _fast_path(int(slots), int(tile)) + base[at:]
    return source.replace(
        f"void {KERNEL}(", f"void __launch_bounds__({THREADS}, {BLOCKS_PER_MULTIPROCESSOR}) {KERNEL}(", 1)


class StencilKernels(NamedTuple):
    """The kernels of a stencil operator (:func:`compiled_kernels`), as ``hold_kernels`` of the operator takes them.

    ``apply`` (``stencil_major_spmm6``) and ``former_recurrence`` are the former kernels.  ``recurrence`` is the
    kernel generated under ``pipelined``, whose fast path takes the launches of ``fast_path_slots`` slots, six
    columns and unit row stride; elsewhere it is ``former_recurrence`` and ``fast_path_slots`` is None.
    ``setting`` is the value of the switch the kernels were compiled for, and ``fallback`` says why ``pipelined``
    holds the former recurrence kernel, where it does.  A plain triple is the former kernels.
    """

    apply: Any
    recurrence: Any
    former_recurrence: Any
    fast_path_slots: int | None = None
    setting: str = "former"
    fallback: str | None = None


class _BuildFailure:
    """What a kernel cache keeps under the key of a generated kernel that could not be built: why."""

    __slots__ = ("reason",)

    def __init__(self, reason):
        self.reason = reason


def _failure_reason(error):
    """One line for the record of a run: the type of ``error`` and the first line of its text, at most 240 characters."""
    lines = str(error).strip().splitlines()
    text = f"{type(error).__name__}: {lines[0].strip()}" if lines else type(error).__name__
    return text if len(text) <= 240 else text[:237] + "..."


def compiled_kernels(cp, cache, lock, key, base, tile, slot_count, setting, compile_raw):
    """The kernels of a stencil operator, :class:`StencilKernels`, compiled once and kept.

    ``base`` is the former source of the layout ``tile`` (0: slot-major) and ``key`` the key under which ``cache``
    has always kept its two kernels (``stencil_major_spmm6`` and ``stencil_major_chebyshev6``); they are compiled
    and kept there under every value of the switch.  ``former`` returns them, its recurrence kernel twice.
    ``pipelined`` compiles :data:`KERNEL` from :func:`pipelined_source` as well, once per device, layout and slot
    count (``key`` with ``setting`` and ``slot_count`` appended), and returns it as the recurrence kernel; the
    apply kernel and the former recurrence kernel, which the launches that do not take the fast path use, are the
    former ones.  Everything is compiled here, when an operator is built, and never while a graph is recorded.
    ``compile_raw`` compiles one ``cp.RawKernel``.

    A generated kernel that cannot be built, because its source or its compilation raises, gives way to the
    former recurrence kernel: the operator is then built with the former kernels, as under ``former``, and
    ``fallback`` says why, with a ``RuntimeWarning``.  The failure is kept under the key of the generated kernel,
    so that the operators of the same device, layout and slot count take the former kernel without compiling
    again.  A failure of the former kernels is raised, as it always was.
    """
    with lock:
        former = cache.get(key)
        if former is None:
            former = tuple(cp.RawKernel(base, name, options=("--std=c++11",))
                           for name in ("stencil_major_spmm6", "stencil_major_chebyshev6"))
            for kernel in former:
                compile_raw(kernel)
            cache[key] = former
        apply_kernel, former_recurrence = former
        if setting == "former":
            return StencilKernels(apply_kernel, former_recurrence, former_recurrence)
        generated = (*key, setting, int(slot_count)) if isinstance(key, tuple) else (key, setting, int(slot_count))
        recurrence = cache.get(generated)
        if recurrence is None:
            try:
                recurrence = cp.RawKernel(pipelined_source(base, slot_count, tile), KERNEL, options=("--std=c++11",))
                compile_raw(recurrence)
            except Exception as error:
                recurrence = _BuildFailure(f"the generated kernel could not be built: {_failure_reason(error)}")
                layout = f"tiles of {int(tile)} rows" if tile else "the slot-major layout"
                try:
                    warnings.warn(
                        f"{SWITCH}={setting}: the recurrence kernel generated for {int(slot_count)} slots in {layout} "
                        f"could not be built ({type(error).__name__}: {error}); the stencil operators of that "
                        "layout and slot count use the former kernels, with the same results",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                except Warning:
                    # A filter that turns warnings into errors is not to turn the former kernels into a failure.
                    pass
            cache[generated] = recurrence
        if isinstance(recurrence, _BuildFailure):
            return StencilKernels(apply_kernel, former_recurrence, former_recurrence, None, setting, recurrence.reason)
        return StencilKernels(apply_kernel, recurrence, former_recurrence, int(slot_count), setting)


__all__ = [
    "DEFAULT", "KERNEL", "KERNELS", "SWITCH", "THREADS", "WIDTH", "StencilKernels", "compiled_kernels",
    "pipelined_source", "stencil_kernel_setting", "takes_fast_path",
]
