// Packing kernels of the libxc evaluators on the device (device_libxc.py), compiled without fused
// multiply-adds: the sums and products of V_xc/pbe.py discrete_gga and discrete_spin_gga_ab in the
// same order.  libxc itself evaluates the functional (its CUDA build, on device pointers).

typedef long long index_t;

// sigma = |D n|^2, summed from zero along x, y, z (pbe.py: sigma += g * g).
#ifdef KERNEL_libxc_sigma
extern "C" __global__ void libxc_sigma(index_t count, const double* gradient, double* sigma) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double g0 = gradient[w], g1 = gradient[count + w], g2 = gradient[2 * count + w];
    double s = 0.0;
    s += g0 * g0;
    s += g1 * g1;
    s += g2 * g2;
    sigma[w] = s;
}
#endif

// libxc's spin layout: rho (count, 2) and sigma (count, 3) = (uu, ud, dd), interleaved per point.
#ifdef KERNEL_libxc_spin_inputs
extern "C" __global__ void libxc_spin_inputs(index_t count, const double* up, const double* down,
                                             const double* gradient_up, const double* gradient_down,
                                             double* rho, double* sigma) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double s_uu = 0.0, s_ud = 0.0, s_dd = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        double gu = gradient_up[axis * count + w];
        double gd = gradient_down[axis * count + w];
        s_uu += gu * gu;
        s_ud += gu * gd;
        s_dd += gd * gd;
    }
    rho[2 * w] = up[w];
    rho[2 * w + 1] = down[w];
    sigma[3 * w] = s_uu;
    sigma[3 * w + 1] = s_ud;
    sigma[3 * w + 2] = s_dd;
}
#endif

// Unpolarized: energy per volume (Hartree) -> Rydberg, the flux 2 vsigma D n, the energy per electron.
#ifdef KERNEL_libxc_outputs
extern "C" __global__ void libxc_outputs(index_t count, const double* density, const double* gradient,
                                         const double* energy, const double* vsigma, double threshold,
                                         double* energy_density, double* flux, double* epsilon) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double e = 2.0 * energy[w];
    double n = density[w];
    energy_density[w] = e;
    double two_vs = 2.0 * vsigma[w];
    flux[w] = two_vs * gradient[w];
    flux[count + w] = two_vs * gradient[count + w];
    flux[2 * count + w] = two_vs * gradient[2 * count + w];
    epsilon[w] = n > threshold ? e / n : 0.0;
}
#endif

// Spin: libxc's interleaved outputs -> channel fields; flux_a = 2 v_aa D n_a + v_ab D n_b.
#ifdef KERNEL_libxc_spin_outputs
extern "C" __global__ void libxc_spin_outputs(index_t count, const double* gradient_up,
                                              const double* gradient_down, const double* energy,
                                              const double* vrho, const double* vsigma,
                                              double* energy_density, double* f_up, double* f_down,
                                              double* flux_up, double* flux_down) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double v_uu = vsigma[3 * w], v_ud = vsigma[3 * w + 1], v_dd = vsigma[3 * w + 2];
    energy_density[w] = 2.0 * energy[w];
    f_up[w] = vrho[2 * w];
    f_down[w] = vrho[2 * w + 1];
    for (int axis = 0; axis < 3; ++axis) {
        double gu = gradient_up[axis * count + w];
        double gd = gradient_down[axis * count + w];
        flux_up[axis * count + w] = 2.0 * v_uu * gu + v_ud * gd;
        flux_down[axis * count + w] = 2.0 * v_dd * gd + v_ud * gu;
    }
}
#endif
