import cupy as cp

# Define global variables for optimization and testing
OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0

_EXC_KERNEL = cp.ElementwiseKernel(
    "T rho, T twovpia0, T p75vpi, T g, T b1, T b2, T c1, T c2, T c3, T c4, T c5",
    "T vxc, T exc_density",
    r"""
    const T zero = (T)0.0;
    const T one = (T)1.0;
    const T four = (T)4.0;
    const T third = (T)(1.0 / 3.0);

    if (rho > zero) {
        const T rs = pow(p75vpi / rho, third);
        T v = -twovpia0 / rs;
        T exc_local = (T)0.75 * rho * v;
        T ec = zero;

        if (rs >= one) {
            const T sqrs = sqrt(rs);
            ec = g / (one + b1 * sqrs + b2 * rs);
            v += (ec * ec) * (one + (T)3.5 * b1 * sqrs * third + four * b2 * rs * third) / g;
        } else {
            const T alpha = log(rs);
            ec = c1 * alpha - c2 + (c3 * alpha - c4) * rs;
            v += ec - (c1 + (c3 * alpha - c5) * rs) * third;
        }

        vxc = v;
        exc_density = exc_local + rho * ec;
    } else {
        vxc = zero;
        exc_density = zero;
    }
    """,
    name="exc_nspn_kernel",
)


def _scalar(value, dtype):
    return dtype(value)


def _kernel_constants(dtype):
    pi = dtype(cp.pi)
    third = dtype(1.0 / 3.0)
    a0 = (dtype(4.0) / (dtype(9.0) * pi)) ** third
    twovpia0 = dtype(2.0) / (pi * a0)
    p75vpi = dtype(0.75) / pi
    return (
        twovpia0,
        p75vpi,
        _scalar(-0.2846, dtype),
        _scalar(1.0529, dtype),
        _scalar(0.3334, dtype),
        _scalar(0.0622, dtype),
        _scalar(0.096, dtype),
        _scalar(0.004, dtype),
        _scalar(0.0232, dtype),
        _scalar(0.0192, dtype),
    )


def warmup(dtype=cp.float32):
    """Compile the fused XC kernel before the first timed call."""
    rho = cp.ones(1, dtype=dtype)
    _EXC_KERNEL(rho, *_kernel_constants(dtype))
    cp.cuda.runtime.deviceSynchronize()


def exc_nspn(Domain, rho, log_target=None):
    """
    GPU version of the Ceperley-Alder LDA exchange-correlation calculation.

    Parameters:
    Domain (object): The domain information including grid spacing.
    rho (array-like): Charge density values on the GPU or host.
    log_target (str or file object): Output file path or handle for logging.

    Returns:
    vxc (cp.ndarray): Exchange-correlation potential on the GPU.
    exc (float): Total exchange-correlation energy as a host scalar.
    """
    rho_gpu = cp.asarray(rho).reshape(-1)
    dtype = cp.float64 if rho_gpu.dtype == cp.float64 else cp.float32
    rho_gpu = rho_gpu.astype(dtype, copy=False)

    dmax = float(cp.max(rho_gpu).item())
    dmin = float(cp.min(rho_gpu).item())

    if log_target:
        try:
            if hasattr(log_target, "write"):
                log_target.write(
                    f" max and min values of charge density [e/bohr^3]   {dmax:.5e}   {dmin:.5e}\n"
                )
            else:
                with open(str(log_target), "a", encoding="utf-8") as fid:
                    fid.write(
                        f" max and min values of charge density [e/bohr^3]   {dmax:.5e}   {dmin:.5e}\n"
                    )
        except OSError:
            pass

    if dmin < 0.0:
        print("Warning: NEGATIVE CHARGE DENSITY FOUND.")

    vxc, exc_density = _EXC_KERNEL(rho_gpu, *_kernel_constants(dtype))
    exc = float((cp.sum(exc_density) * (Domain["h"] ** 3)).item())
    return vxc, exc
