// Exchange-correlation functionals on a CUDA device, in float64.
//
// Every function below is a line-by-line translation of the readable NumPy
// functional of parsec_python.V_xc it names: the same operations on the same
// operands in the same order.  The module is compiled with --fmad=false, so
// each product and sum is rounded on its own as NumPy rounds it; only the
// elementary functions (cbrt, pow, log, log1p, exp, expm1, asinh, sqrt) are
// CUDA's, within an ulp or two of the host library.  The constants between
// at signs are written in by device_xc.py from the Python modules.
//
// Grid geometry.  A value is one physical density value: a grid row without
// symmetry, one orbit of a reflection group with it.  ``value_of_box`` maps
// every point of the Cartesian box of the grid (row-major, lengths n0 n1 n2)
// to its value, -1 outside the cluster domain; ``positions`` holds the box
// index of the representative point of every value.  A periodic grid is its
// own box, and its stencil wraps around.  ``flips`` (bit a set where the
// reflection that carries the representative onto the box point reverses
// axis a) carries the components of a vector field, the GGA flux, from the
// representative to the other points of its orbit.
//
// A module holds the kernels named by -DKERNEL_<name> options only (each
// functional compiles what it launches).
//
// Stencils.  The first derivative along an axis is that of
// scipy.ndimage.correlate1d with an antisymmetric filter of 2M+1 weights,
// which pairs the samples:  out = in[0] w[M] + sum_{j=-M..-1} (in[j] - in[-j]) w[M+j].

#define PI @PI@
#define DENSITY_THRESHOLD @DENSITY_THRESHOLD@
#define KF @KF@
// V_xc/pbe.py
#define PBE_KAPPA @PBE_KAPPA@
#define PBE_MU @PBE_MU@
#define PBE_MU_OVER_KAPPA @PBE_MU_OVER_KAPPA@
#define PBE_BETA @PBE_BETA@
#define PBE_GAMMA @PBE_GAMMA@
#define PBE_BETA_OVER_GAMMA @PBE_BETA_OVER_GAMMA@
#define PBE_CX @PBE_CX@
// V_xc/pbe_spin.py
#define SPIN_GAM @SPIN_GAM@
#define SPIN_FZZ @SPIN_FZZ@
#define SPIN_ETA @SPIN_ETA@
// V_xc/pw92_lda.py
#define PW_AX @PW_AX@
#define PW_GAM @PW_GAM@
#define PW_FZZ @PW_FZZ@
#define PW_A_UNPOLARIZED @PW_A_UNPOLARIZED@
#define TWO_OVER_PI_A0 @TWO_OVER_PI_A0@
#define THREE_QUARTERS_OVER_PI @THREE_QUARTERS_OVER_PI@
// V_xc/pw91.py
#define PW91_A @PW91_A@
#define PW91_A1 @PW91_A1@
#define PW91_A2 @PW91_A2@
#define PW91_A3 @PW91_A3@
#define PW91_A4 @PW91_A4@
#define PW91_B1 @PW91_B1@
#define PW91_XNU @PW91_XNU@
#define PW91_CC0 @PW91_CC0@
#define PW91_CX @PW91_CX@
#define PW91_C1 @PW91_C1@
#define PW91_C2 @PW91_C2@
#define PW91_C3 @PW91_C3@
#define PW91_C4 @PW91_C4@
#define PW91_C5 @PW91_C5@
#define PW91_C6 @PW91_C6@
#define PW91_H1_EXPONENT @PW91_H1_EXPONENT@
#define PW91_KS2_OVER_KF2 @PW91_KS2_OVER_KF2@
#define PW91_BET @PW91_BET@
#define PW91_DELT @PW91_DELT@
#define PW91_ETA @PW91_ETA@
// V_xc/local_density.py
#define XALPHA_ALPHA @XALPHA_ALPHA@
#define WIGNER_A @WIGNER_A@
#define WIGNER_B @WIGNER_B@
#define HL_INVERSE_A @HL_INVERSE_A@
#define HL_C @HL_C@
// V_xc/ca_lda.py
#define CA_PI_A0 @CA_PI_A0@
// V_xc/ca_lda_spin.py
#define CA_GAMMA @CA_GAMMA@
// V_xc/asymptotic_correction.py
#define LB_TWO_BETA @LB_TWO_BETA@
#define LB_THREE_BETA @LB_THREE_BETA@

typedef long long index_t;

// ---------------------------------------------------------------------------
// Stencils on the value map
// ---------------------------------------------------------------------------

struct BoxPoint {
    index_t position;
    int coordinate[3];
};

static __device__ __forceinline__ BoxPoint box_point(index_t position, int n1, int n2) {
    BoxPoint point;
    point.position = position;
    point.coordinate[2] = (int)(position % n2);
    point.coordinate[1] = (int)((position / n2) % n1);
    point.coordinate[0] = (int)(position / ((index_t)n1 * n2));
    return point;
}

// Box index of the point ``offset`` steps along ``axis``, -1 outside the box.
static __device__ __forceinline__ index_t box_neighbor(const BoxPoint& point, int axis, int offset,
                                                const int* lengths, const index_t* strides,
                                                int periodic) {
    int coordinate = point.coordinate[axis] + offset;
    if (periodic) {
        coordinate %= lengths[axis];
        if (coordinate < 0) coordinate += lengths[axis];
    } else if (coordinate < 0 || coordinate >= lengths[axis]) {
        return -1;
    }
    return point.position + (index_t)(coordinate - point.coordinate[axis]) * strides[axis];
}

static __device__ __forceinline__ double scalar_sample(const double* values, const int* value_of_box,
                                                index_t box) {
    if (box < 0) return 0.0;
    int value = value_of_box[box];
    return value < 0 ? 0.0 : values[value];
}

// Component ``axis`` of a vector field stored at the representatives.
static __device__ __forceinline__ double vector_sample(const double* component, const int* value_of_box,
                                                const unsigned char* flips, int use_flips, index_t box,
                                                int axis) {
    if (box < 0) return 0.0;
    int value = value_of_box[box];
    if (value < 0) return 0.0;
    double sample = component[value];
    if (use_flips && ((flips[box] >> axis) & 1)) sample = -sample;
    return sample;
}

// pbe.py density_gradient / discrete_gga: (D f)_a at every value, 3 x count.
#ifdef KERNEL_xc_gradient
extern "C" __global__ void xc_gradient(
    index_t count, const double* values, const int* value_of_box, const int* positions,
    int n0, int n1, int n2, int periodic, int width, const double* weights, double* gradient) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    const int lengths[3] = {n0, n1, n2};
    const index_t strides[3] = {(index_t)n1 * n2, (index_t)n2, (index_t)1};
    BoxPoint point = box_point((index_t)positions[w], n1, n2);
    double center = values[w];
    for (int axis = 0; axis < 3; ++axis) {
        double sum = center * weights[width];
        for (int j = -width; j < 0; ++j) {
            double left = scalar_sample(values, value_of_box,
                                        box_neighbor(point, axis, j, lengths, strides, periodic));
            double right = scalar_sample(values, value_of_box,
                                         box_neighbor(point, axis, -j, lengths, strides, periodic));
            sum += (left - right) * weights[width + j];
        }
        gradient[axis * count + w] = sum;
    }
}
#endif

// out = scale * (base + sum_a D_a^T flux_a), the transpose applied as the
// correlation with the negated weights, axis after axis (pbe.py discrete_gga).
#ifdef KERNEL_xc_adjoint
extern "C" __global__ void xc_adjoint(
    index_t count, const double* flux, const int* value_of_box, const unsigned char* flips,
    int use_flips, const int* positions, int n0, int n1, int n2, int periodic, int width,
    const double* negated_weights, const double* base, double scale, double* out) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    const int lengths[3] = {n0, n1, n2};
    const index_t strides[3] = {(index_t)n1 * n2, (index_t)n2, (index_t)1};
    BoxPoint point = box_point((index_t)positions[w], n1, n2);
    double total = base[w];
    for (int axis = 0; axis < 3; ++axis) {
        const double* component = flux + axis * count;
        double sum = component[w] * negated_weights[width];
        for (int j = -width; j < 0; ++j) {
            double left = vector_sample(component, value_of_box, flips, use_flips,
                                        box_neighbor(point, axis, j, lengths, strides, periodic), axis);
            double right = vector_sample(component, value_of_box, flips, use_flips,
                                         box_neighbor(point, axis, -j, lengths, strides, periodic), axis);
            sum += (left - right) * negated_weights[width + j];
        }
        total += sum;
    }
    out[w] = scale * total;
}
#endif

// ---------------------------------------------------------------------------
// Perdew-Wang pieces (pbe.py _pw92_g, pbe_spin.py pw92_spin_correlation,
// pw92_lda.py pw92_lsd_correlation)
// ---------------------------------------------------------------------------

static __device__ void pw92_g(double rs, double a, double alpha1, double beta1, double beta2,
                       double beta3, double beta4, double* epsilon, double* derivative) {
    double sqrt_rs = sqrt(rs);
    double polynomial = beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs + beta4 * rs * rs;
    double q = 2.0 * a * polynomial;
    double log_term = log1p(1.0 / q);
    *epsilon = -2.0 * a * (1.0 + alpha1 * rs) * log_term;
    double polynomial_prime = 0.5 * beta1 / sqrt_rs + beta2 + 1.5 * beta3 * sqrt_rs + 2.0 * beta4 * rs;
    double q_prime = 2.0 * a * polynomial_prime;
    double log_prime = -q_prime / (q * (q + 1.0));
    *derivative = -2.0 * a * (alpha1 * log_term + (1.0 + alpha1 * rs) * log_prime);
}

// e_c(r_s, zeta), d/dr_s, d/dzeta with the unpolarized A, gamma and f''(0)
// of the caller: PBE's (pbe_spin.py) or Fortran's (pw92_lda.py).
static __device__ void pw92_lsd(double rs, double zeta, double a_unpolarized, double gam, double fzz,
                         double* energy, double* energy_rs, double* energy_zeta) {
    double unpolarized, unpolarized_rs, polarized, polarized_rs, stiffness, stiffness_rs;
    pw92_g(rs, a_unpolarized, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, &unpolarized, &unpolarized_rs);
    pw92_g(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, &polarized, &polarized_rs);
    pw92_g(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, &stiffness, &stiffness_rs);
    double plus = fmax(1.0 + zeta, 0.0);
    double minus = fmax(1.0 - zeta, 0.0);
    double z4 = pow(zeta, 4.0);
    double f = (pow(plus, 4.0 / 3.0) + pow(minus, 4.0 / 3.0) - 2.0) / gam;
    *energy = unpolarized * (1.0 - f * z4) + polarized * f * z4 - stiffness * f * (1.0 - z4) / fzz;
    *energy_rs = unpolarized_rs * (1.0 - f * z4) + polarized_rs * f * z4
                 - stiffness_rs * f * (1.0 - z4) / fzz;
    double f_zeta = (4.0 / 3.0) * (cbrt(plus) - cbrt(minus)) / gam;
    *energy_zeta = 4.0 * pow(zeta, 3.0) * f * (polarized - unpolarized + stiffness / fzz)
                   + f_zeta * (z4 * polarized - z4 * unpolarized - (1.0 - z4) * stiffness / fzz);
}

// ---------------------------------------------------------------------------
// PBE (pbe.py pbe_energy_partials, pbe_spin.py pbe_spin_energy_partials)
// ---------------------------------------------------------------------------

static __device__ void pbe_exchange(double n, double sig, double* energy, double* d_n, double* d_sigma) {
    double n13 = cbrt(n);
    double n43 = n * n13;
    double kf = KF * n13;
    double s2 = sig / (4.0 * kf * kf * n * n);
    double exchange_denominator = 1.0 + PBE_MU_OVER_KAPPA * s2;
    double enhancement = 1.0 + PBE_KAPPA - PBE_KAPPA / exchange_denominator;
    double enhancement_prime = PBE_MU / (exchange_denominator * exchange_denominator);
    *energy = -PBE_CX * n43 * enhancement;
    *d_n = -PBE_CX * n13 * ((4.0 / 3.0) * enhancement - (8.0 / 3.0) * s2 * enhancement_prime);
    double ds2_dsigma = 1.0 / (4.0 * kf * kf * n * n);
    *d_sigma = -PBE_CX * n43 * enhancement_prime * ds2_dsigma;
}

static __device__ void pbe_unpolarized(double n, double sigma, double* f, double* f_n, double* f_sigma) {
    *f = 0.0;
    *f_n = 0.0;
    *f_sigma = 0.0;
    if (!(n > DENSITY_THRESHOLD)) return;
    double sig = fmax(sigma, 0.0);
    double n13 = cbrt(n);
    double kf = KF * n13;
    double exchange_energy, exchange_dn, exchange_dsigma;
    pbe_exchange(n, sig, &exchange_energy, &exchange_dn, &exchange_dsigma);

    double rs = cbrt(3.0 / (4.0 * PI * n));
    double epsilon_lda, epsilon_lda_rs;
    pw92_g(rs, PBE_GAMMA, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, &epsilon_lda, &epsilon_lda_rs);
    double epsilon_lda_n = -rs * epsilon_lda_rs / (3.0 * n);

    double beta_over_gamma = PBE_BETA_OVER_GAMMA;
    double exponential_minus_one = expm1(-epsilon_lda / PBE_GAMMA);
    double a_parameter = beta_over_gamma / exponential_minus_one;
    double a_epsilon = beta_over_gamma * exp(-epsilon_lda / PBE_GAMMA)
                       / (PBE_GAMMA * exponential_minus_one * exponential_minus_one);

    double dt2_dsigma = PI / (16.0 * kf * n * n);
    double t2 = sig * dt2_dsigma;
    double at = a_parameter * t2;
    double numerator = beta_over_gamma * t2 * (1.0 + at);
    double denominator = 1.0 + at + at * at;
    double q = numerator / denominator;
    double h = PBE_GAMMA * log1p(q);

    double numerator_t = beta_over_gamma * (1.0 + 2.0 * at);
    double denominator_t = a_parameter + 2.0 * a_parameter * at;
    double q_t = (numerator_t * denominator - numerator * denominator_t) / (denominator * denominator);
    double numerator_a = beta_over_gamma * t2 * t2;
    double denominator_a = t2 + 2.0 * a_parameter * t2 * t2;
    double q_a = (numerator_a * denominator - numerator * denominator_a) / (denominator * denominator);
    double h_q = PBE_GAMMA / (1.0 + q);
    double h_t = h_q * q_t;
    double h_a = h_q * q_a;

    double t2_n = -(7.0 / 3.0) * t2 / n;
    double h_n = h_t * t2_n + h_a * a_epsilon * epsilon_lda_n;
    double h_sigma = h_t * dt2_dsigma;
    double correlation_energy = n * (epsilon_lda + h);
    double correlation_dn = epsilon_lda + h + n * (epsilon_lda_n + h_n);
    double correlation_dsigma = n * h_sigma;

    *f = exchange_energy + correlation_energy;
    *f_n = exchange_dn + correlation_dn;
    *f_sigma = exchange_dsigma + correlation_dsigma;
}

// out: f, df/dn_up, df/dn_down, df/dsigma_uu, df/dsigma_dd, df/dsigma.
static __device__ void pbe_spin(double density_up, double density_down, double sig_up, double sig_down,
                         double sig_total, double* out) {
    double up = fmax(density_up, 0.0);
    double down = fmax(density_down, 0.0);
    double energy = 0.0, d_up = 0.0, d_down = 0.0, d_sigma_up = 0.0, d_sigma_down = 0.0;
    double d_sigma_total = 0.0;
    double doubled = 2.0 * up;
    if (doubled > DENSITY_THRESHOLD) {
        double exchange, exchange_dn, exchange_dsigma;
        pbe_exchange(doubled, fmax(4.0 * sig_up, 0.0), &exchange, &exchange_dn, &exchange_dsigma);
        energy += 0.5 * exchange;
        d_up += exchange_dn;
        d_sigma_up += 2.0 * exchange_dsigma;
    }
    doubled = 2.0 * down;
    if (doubled > DENSITY_THRESHOLD) {
        double exchange, exchange_dn, exchange_dsigma;
        pbe_exchange(doubled, fmax(4.0 * sig_down, 0.0), &exchange, &exchange_dn, &exchange_dsigma);
        energy += 0.5 * exchange;
        d_down += exchange_dn;
        d_sigma_down += 2.0 * exchange_dsigma;
    }
    double density = up + down;
    if (density > DENSITY_THRESHOLD) {
        double n = density;
        double sig = fmax(sig_total, 0.0);
        double zeta = (up - down) / n;
        double kf = KF * cbrt(n);
        double rs = cbrt(3.0 / (4.0 * PI * n));
        double epsilon, epsilon_rs, epsilon_zeta;
        pw92_lsd(rs, zeta, PBE_GAMMA, SPIN_GAM, SPIN_FZZ, &epsilon, &epsilon_rs, &epsilon_zeta);
        double epsilon_n = -rs * epsilon_rs / (3.0 * n);

        double plus = fmax(1.0 + zeta, 0.0);
        double minus = fmax(1.0 - zeta, 0.0);
        double phi = 0.5 * (pow(plus, 2.0 / 3.0) + pow(minus, 2.0 / 3.0));
        double one_plus = 1.0 + zeta;
        double one_minus = 1.0 - zeta;
        double phi_zeta = (pow(one_plus * one_plus + SPIN_ETA, -1.0 / 6.0)
                           - pow(one_minus * one_minus + SPIN_ETA, -1.0 / 6.0)) / 3.0;
        double phi3 = phi * phi * phi;

        double beta_over_gamma = PBE_BETA_OVER_GAMMA;
        double exponent = -epsilon / (PBE_GAMMA * phi3);
        double exponential_minus_one = expm1(exponent);
        double a_parameter = beta_over_gamma / exponential_minus_one;
        double a_exponent = -beta_over_gamma * exp(exponent)
                            / (exponential_minus_one * exponential_minus_one);
        double a_epsilon = -a_exponent / (PBE_GAMMA * phi3);
        double a_phi = a_exponent * 3.0 * epsilon / (PBE_GAMMA * phi3 * phi);

        double dt2_dsigma = PI / (16.0 * phi * phi * kf * n * n);
        double t2 = sig * dt2_dsigma;
        double at = a_parameter * t2;
        double numerator = beta_over_gamma * t2 * (1.0 + at);
        double denominator = 1.0 + at + at * at;
        double q = numerator / denominator;
        double log_term = log1p(q);
        double h = PBE_GAMMA * phi3 * log_term;

        double numerator_t = beta_over_gamma * (1.0 + 2.0 * at);
        double denominator_t = a_parameter + 2.0 * a_parameter * at;
        double q_t = (numerator_t * denominator - numerator * denominator_t) / (denominator * denominator);
        double numerator_a = beta_over_gamma * t2 * t2;
        double denominator_a = t2 + 2.0 * a_parameter * t2 * t2;
        double q_a = (numerator_a * denominator - numerator * denominator_a) / (denominator * denominator);
        double h_q = PBE_GAMMA * phi3 / (1.0 + q);
        double h_t = h_q * q_t;
        double h_a = h_q * q_a;
        double h_phi = 3.0 * PBE_GAMMA * phi * phi * log_term + h_a * a_phi - h_t * 2.0 * t2 / phi;

        double t2_n = -(7.0 / 3.0) * t2 / n;
        double per_electron = epsilon + h;
        double per_electron_n = epsilon_n + h_t * t2_n + h_a * a_epsilon * epsilon_n;
        double per_electron_zeta = epsilon_zeta + h_a * a_epsilon * epsilon_zeta + h_phi * phi_zeta;
        double common = per_electron + n * per_electron_n;

        energy += n * per_electron;
        d_up += common + (1.0 - zeta) * per_electron_zeta;
        d_down += common - (1.0 + zeta) * per_electron_zeta;
        d_sigma_total = n * h_t * dt2_dsigma;
    }
    out[0] = energy;
    out[1] = d_up;
    out[2] = d_down;
    out[3] = d_sigma_up;
    out[4] = d_sigma_down;
    out[5] = d_sigma_total;
}

// ---------------------------------------------------------------------------
// PW91 (pw91.py)
// ---------------------------------------------------------------------------

static __device__ void pw91_exchange(double n, double sig, double* energy, double* d_n, double* d_sigma) {
    double n13 = cbrt(n);
    double kf = KF * n13;
    double scale = 1.0 / (4.0 * kf * kf * n * n);
    double s2 = sig * scale;
    double s = sqrt(s2);
    double p0 = 1.0 / sqrt(1.0 + PW91_A * PW91_A * s2);
    double p1 = asinh(PW91_A * s);
    double p2 = exp(-PW91_A4 * s2);
    double p3 = 1.0 / (1.0 + PW91_A1 * s * p1 + PW91_B1 * s2 * s2);
    double p4 = 1.0 + PW91_A1 * s * p1 + (PW91_A2 - PW91_A3 * p2) * s2;
    double enhancement = p3 * p4;
    double p5 = PW91_B1 * s2 - (PW91_A2 - PW91_A3 * p2);
    double p6 = PW91_A1 * s * (p1 + PW91_A * s * p0);
    double p7 = 2.0 * (PW91_A2 - PW91_A3 * p2) + 2.0 * PW91_A3 * PW91_A4 * s2 * p2
                - 4.0 * PW91_B1 * s2 * enhancement;
    double fs = p3 * (p3 * p5 * p6 + p7);
    double prefactor = PW_AX * n13;
    *energy = n * prefactor * enhancement;
    *d_n = (4.0 / 3.0) * prefactor * (enhancement - s2 * fs);
    *d_sigma = 0.5 * n * prefactor * fs * scale;
}

// out: H, hrs, ht, hz of pw91.py _gradient_correction.
static __device__ void pw91_gradient_correction(double rs, double g, double gz, double ec, double ecrs,
                                         double eczet, double t2, double* out) {
    double g3 = g * g * g;
    double g4 = g3 * g;
    double pon = -PW91_DELT * ec / (g3 * PW91_BET);
    double b = PW91_DELT / expm1(pon);
    double b2 = b * b;
    double t4 = t2 * t2;
    double t6 = t4 * t2;
    double q4 = 1.0 + b * t2;
    double q5 = 1.0 + b * t2 + b2 * t4;
    double q6 = PW91_C1 + PW91_C2 * rs + PW91_C3 * rs * rs;
    double q7 = 1.0 + PW91_C4 * rs + PW91_C5 * rs * rs + PW91_C6 * rs * rs * rs;
    double cc = -PW91_CX + q6 / q7;
    double r1 = PW91_H1_EXPONENT * PW91_KS2_OVER_KF2 * rs * g4;
    double coeff = cc - PW91_CC0 - 3.0 * PW91_CX / 7.0;
    double r2 = PW91_XNU * coeff * g3;
    double r3 = exp(-r1 * t2);
    double h0 = g3 * (PW91_BET / PW91_DELT) * log1p(PW91_DELT * q4 * t2 / q5);
    double h1 = r3 * r2 * t2;

    double ccrs = (PW91_C2 + 2.0 * PW91_C3 * rs) / q7
                  - q6 * (PW91_C4 + 2.0 * PW91_C5 * rs + 3.0 * PW91_C6 * rs * rs) / (q7 * q7);
    double rsthrd = rs / 3.0;
    double r4 = rsthrd * ccrs / coeff;
    double fac = PW91_DELT / b + 1.0;
    double bg = -3.0 * b2 * ec * fac / (PW91_BET * g4);
    double bec = b2 * fac / (PW91_BET * g3);
    double q8 = q5 * q5 + PW91_DELT * q4 * q5 * t2;
    double q9 = 1.0 + 2.0 * b * t2;
    double h0b = -PW91_BET * g3 * b * t6 * (2.0 + b * t2) / q8;
    double h0rs = -rsthrd * h0b * bec * ecrs;
    double h0z = 3.0 * gz * h0 / g + h0b * (bg * gz + bec * eczet);
    double h0t = 2.0 * PW91_BET * g3 * q9 / q8;
    double h1rs = r3 * r2 * t2 * (-r4 + r1 * t2 / 3.0);
    double h1z = gz * r3 * r2 * t2 * (3.0 - 4.0 * r1 * t2) / g;
    double h1t = 2.0 * r3 * r2 * (1.0 - r1 * t2);
    out[0] = h0 + h1;
    out[1] = h0rs + h1rs;
    out[2] = h0t + h1t;
    out[3] = h0z + h1z;
}

static __device__ void pw91_unpolarized(double n, double sigma, double* f, double* f_n, double* f_sigma) {
    *f = 0.0;
    *f_n = 0.0;
    *f_sigma = 0.0;
    if (!(n > DENSITY_THRESHOLD)) return;
    double sig = fmax(sigma, 0.0);
    double exchange, exchange_dn, exchange_dsigma;
    pw91_exchange(n, sig, &exchange, &exchange_dn, &exchange_dsigma);
    double kf = KF * cbrt(n);
    double rs = cbrt(3.0 / (4.0 * PI * n));
    double zeta = 0.0;
    double ec, ecrs, eczet;
    pw92_lsd(rs, zeta, PW_A_UNPOLARIZED, PW_GAM, PW_FZZ, &ec, &ecrs, &eczet);
    double dt2_dsigma = PI / (16.0 * kf * n * n);
    double t2 = sig * dt2_dsigma;
    double correction[4];
    pw91_gradient_correction(rs, 1.0, zeta, ec, ecrs, eczet, t2, correction);
    double h = correction[0], hrs = correction[1], ht = correction[2];
    *f = exchange + n * (ec + h);
    *f_n = exchange_dn + (ec - rs * ecrs / 3.0 + h + hrs - (7.0 / 6.0) * t2 * ht);
    *f_sigma = exchange_dsigma + 0.5 * n * ht * dt2_dsigma;
}

static __device__ void pw91_spin(double density_up, double density_down, double sig_up, double sig_down,
                          double sig_total, double* out) {
    double up = fmax(density_up, 0.0);
    double down = fmax(density_down, 0.0);
    double energy = 0.0, d_up = 0.0, d_down = 0.0, d_sigma_up = 0.0, d_sigma_down = 0.0;
    double d_sigma_total = 0.0;
    double doubled = 2.0 * up;
    if (doubled > DENSITY_THRESHOLD) {
        double exchange, exchange_dn, exchange_dsigma;
        pw91_exchange(doubled, fmax(4.0 * sig_up, 0.0), &exchange, &exchange_dn, &exchange_dsigma);
        energy += 0.5 * exchange;
        d_up += exchange_dn;
        d_sigma_up += 2.0 * exchange_dsigma;
    }
    doubled = 2.0 * down;
    if (doubled > DENSITY_THRESHOLD) {
        double exchange, exchange_dn, exchange_dsigma;
        pw91_exchange(doubled, fmax(4.0 * sig_down, 0.0), &exchange, &exchange_dn, &exchange_dsigma);
        energy += 0.5 * exchange;
        d_down += exchange_dn;
        d_sigma_down += 2.0 * exchange_dsigma;
    }
    double density = up + down;
    if (density > DENSITY_THRESHOLD) {
        double n = density;
        double sig = fmax(sig_total, 0.0);
        double zeta = (up - down) / n;
        double kf = KF * cbrt(n);
        double rs = cbrt(3.0 / (4.0 * PI * n));
        double ec, ecrs, eczet;
        pw92_lsd(rs, zeta, PW_A_UNPOLARIZED, PW_GAM, PW_FZZ, &ec, &ecrs, &eczet);
        double plus = fmax(1.0 + zeta, 0.0);
        double minus = fmax(1.0 - zeta, 0.0);
        double g = 0.5 * (pow(plus, 2.0 / 3.0) + pow(minus, 2.0 / 3.0));
        double one_plus = 1.0 + zeta;
        double one_minus = 1.0 - zeta;
        double gz = (pow(one_plus * one_plus + PW91_ETA, -1.0 / 6.0)
                     - pow(one_minus * one_minus + PW91_ETA, -1.0 / 6.0)) / 3.0;
        double dt2_dsigma = PI / (16.0 * g * g * kf * n * n);
        double t2 = sig * dt2_dsigma;
        double correction[4];
        pw91_gradient_correction(rs, g, gz, ec, ecrs, eczet, t2, correction);
        double h = correction[0], hrs = correction[1], ht = correction[2], hz = correction[3];
        double pref = hz - gz * t2 * ht / g;
        double common = ec - rs * ecrs / 3.0 + h + hrs - (7.0 / 6.0) * t2 * ht;
        energy += n * (ec + h);
        d_up += common + (1.0 - zeta) * (eczet + pref);
        d_down += common - (1.0 + zeta) * (eczet + pref);
        d_sigma_total = 0.5 * n * ht * dt2_dsigma;
    }
    out[0] = energy;
    out[1] = d_up;
    out[2] = d_down;
    out[3] = d_sigma_up;
    out[4] = d_sigma_down;
    out[5] = d_sigma_total;
}

// ---------------------------------------------------------------------------
// GGA kernels: partials at every value, energy density (Ry), the flux
// 2 f_sigma D n (Hartree) and the energy per electron (Ry)
// ---------------------------------------------------------------------------

#define GGA_UNPOLARIZED_KERNEL(NAME, PARTIALS)                                                 \
    extern "C" __global__ void NAME(index_t count, const double* density, const double* gradient, \
                                    double* energy_density, double* f_n, double* flux,          \
                                    double* epsilon) {                                          \
        index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;                             \
        if (w >= count) return;                                                                 \
        double g0 = gradient[w], g1 = gradient[count + w], g2 = gradient[2 * count + w];        \
        double sigma = 0.0;                                                                     \
        sigma += g0 * g0;                                                                       \
        sigma += g1 * g1;                                                                       \
        sigma += g2 * g2;                                                                       \
        double n = density[w];                                                                  \
        double f, fn, fs;                                                                       \
        PARTIALS(n, sigma, &f, &fn, &fs);                                                       \
        double energy = 2.0 * f;                                                                \
        energy_density[w] = energy;                                                             \
        f_n[w] = fn;                                                                            \
        flux[w] = 2.0 * fs * g0;                                                                \
        flux[count + w] = 2.0 * fs * g1;                                                        \
        flux[2 * count + w] = 2.0 * fs * g2;                                                    \
        epsilon[w] = n > DENSITY_THRESHOLD ? energy / n : 0.0;                                   \
    }

#ifdef KERNEL_gga_pbe
GGA_UNPOLARIZED_KERNEL(gga_pbe, pbe_unpolarized)
#endif
#ifdef KERNEL_gga_pw91
GGA_UNPOLARIZED_KERNEL(gga_pw91, pw91_unpolarized)
#endif

// pbe_spin.py discrete_spin_gga: the flux of channel s is
// 2 f_sigma_ss D n_s + 2 f_sigma D n with D n = D n_up + D n_down.
#define GGA_SPIN_KERNEL(NAME, PARTIALS)                                                         \
    extern "C" __global__ void NAME(index_t count, const double* up, const double* down,        \
                                    const double* gradient_up, const double* gradient_down,     \
                                    double* energy_density, double* f_up, double* f_down,       \
                                    double* flux_up, double* flux_down) {                       \
        index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;                             \
        if (w >= count) return;                                                                 \
        double gu[3], gd[3], gt[3];                                                             \
        for (int axis = 0; axis < 3; ++axis) {                                                  \
            gu[axis] = gradient_up[axis * count + w];                                           \
            gd[axis] = gradient_down[axis * count + w];                                         \
            gt[axis] = gu[axis] + gd[axis];                                                     \
        }                                                                                       \
        double s_up = gu[0] * gu[0] + gu[1] * gu[1] + gu[2] * gu[2];                            \
        double s_down = gd[0] * gd[0] + gd[1] * gd[1] + gd[2] * gd[2];                          \
        double s_total = gt[0] * gt[0] + gt[1] * gt[1] + gt[2] * gt[2];                         \
        double out[6];                                                                          \
        PARTIALS(up[w], down[w], s_up, s_down, s_total, out);                                   \
        energy_density[w] = 2.0 * out[0];                                                       \
        f_up[w] = out[1];                                                                       \
        f_down[w] = out[2];                                                                     \
        for (int axis = 0; axis < 3; ++axis) {                                                  \
            flux_up[axis * count + w] = 2.0 * out[3] * gu[axis] + 2.0 * out[5] * gt[axis];      \
            flux_down[axis * count + w] = 2.0 * out[4] * gd[axis] + 2.0 * out[5] * gt[axis];    \
        }                                                                                       \
    }

#ifdef KERNEL_spin_gga_pbe
GGA_SPIN_KERNEL(spin_gga_pbe, pbe_spin)
#endif
#ifdef KERNEL_spin_gga_pw91
GGA_SPIN_KERNEL(spin_gga_pw91, pw91_spin)
#endif

// ---------------------------------------------------------------------------
// Local density functionals (pw92_lda.py, local_density.py), Rydberg
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double dirac_exchange_potential(double rs) {
    return -TWO_OVER_PI_A0 / rs;
}

static __device__ void pw92_point(double rs, double* potential, double* energy) {
    // pw92_lda: V = v_x + 2 (e_c - r_s e_c'/3), e = 3/4 v_x + 2 e_c (per electron).
    double exchange = dirac_exchange_potential(rs);
    double correlation, correlation_rs;
    pw92_g(rs, PW_A_UNPOLARIZED, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, &correlation, &correlation_rs);
    *potential = exchange + 2.0 * (correlation - rs * correlation_rs / 3.0);
    *energy = 0.75 * exchange + 2.0 * correlation;
}

static __device__ void xalpha_point(double rs, double* potential, double* energy) {
    *potential = 1.5 * XALPHA_ALPHA * dirac_exchange_potential(rs);
    *energy = 0.75 * *potential;
}

static __device__ void wigner_point(double rs, double* potential, double* energy) {
    double exchange = dirac_exchange_potential(rs);
    double shifted = rs + WIGNER_B;
    *potential = exchange + WIGNER_A * (4.0 * rs / 3.0 + WIGNER_B) / (shifted * shifted);
    *energy = 0.75 * exchange + WIGNER_A / shifted;
}

static __device__ void hedin_lundqvist_point(double rs, double* potential, double* energy) {
    double exchange = dirac_exchange_potential(rs);
    double x = rs * HL_INVERSE_A;
    double logarithm = log1p(1.0 / x);
    *potential = exchange - HL_C * logarithm;
    *energy = 0.75 * exchange - HL_C * ((1.0 + x * x * x) * logarithm + 0.5 * x - x * x - 1.0 / 3.0);
}

// pw92_lda.py: the energy density is rho times the energy per electron; the
// latter is energy_density / density where the density is positive.
#define LDA_KERNEL(NAME, POINT)                                                                 \
    extern "C" __global__ void NAME(index_t count, const double* density, double* potential,    \
                                    double* energy_density, double* epsilon) {                  \
        index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;                             \
        if (w >= count) return;                                                                 \
        double rho = density[w];                                                                \
        double v = 0.0, e = 0.0;                                                                \
        if (rho > 0.0) {                                                                        \
            double rs = cbrt(THREE_QUARTERS_OVER_PI / rho);                                      \
            double point_energy;                                                                \
            POINT(rs, &v, &point_energy);                                                       \
            e = rho * point_energy;                                                             \
        }                                                                                       \
        potential[w] = v;                                                                       \
        energy_density[w] = e;                                                                  \
        epsilon[w] = rho > 0.0 ? e / rho : 0.0;                                                 \
    }

#ifdef KERNEL_lda_pw92
LDA_KERNEL(lda_pw92, pw92_point)
#endif
#ifdef KERNEL_lda_xalpha
LDA_KERNEL(lda_xalpha, xalpha_point)
#endif
#ifdef KERNEL_lda_wigner
LDA_KERNEL(lda_wigner, wigner_point)
#endif
#ifdef KERNEL_lda_hedin_lundqvist
LDA_KERNEL(lda_hedin_lundqvist, hedin_lundqvist_point)
#endif

// pw92_lda.py pw92_lda_spin; ``up`` and ``down`` hold their halves of the core.
#ifdef KERNEL_lda_spin_pw92
extern "C" __global__ void lda_spin_pw92(index_t count, const double* up, const double* down,
                                         double* potential_up, double* potential_down,
                                         double* energy_density) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double rho_up = up[w], rho_down = down[w];
    double rho = rho_up + rho_down;
    double vu = 0.0, vd = 0.0, e = 0.0;
    if (rho > 0.0) {
        double exchange_up = PW_AX * cbrt(2.0 * rho_up);
        double exchange_down = PW_AX * cbrt(2.0 * rho_down);
        double rs = cbrt(THREE_QUARTERS_OVER_PI / rho);
        double zeta = (rho_up - rho_down) / rho;
        double correlation, correlation_rs, correlation_zeta;
        pw92_lsd(rs, zeta, PW_A_UNPOLARIZED, PW_GAM, PW_FZZ, &correlation, &correlation_rs,
                 &correlation_zeta);
        double common = correlation - rs * correlation_rs / 3.0 - zeta * correlation_zeta;
        vu = 2.0 * ((4.0 / 3.0) * exchange_up + common + correlation_zeta);
        vd = 2.0 * ((4.0 / 3.0) * exchange_down + common - correlation_zeta);
        e = 2.0 * (rho_up * exchange_up + rho_down * exchange_down + correlation * rho);
    }
    potential_up[w] = vu;
    potential_down[w] = vd;
    energy_density[w] = e;
}
#endif

// ---------------------------------------------------------------------------
// CA/PZ LDA (ca_lda.py ca_lda, ca_lda_spin.py ca_lda_spin) and the LB94 /
// Casida-Salahub corrections (asymptotic_correction.py)
// ---------------------------------------------------------------------------

// ca_lda.py for one positive density: V_xc and epsilon_xc (Rydberg).
static __device__ void ca_lda_point(double rho, double* potential, double* epsilon) {
    double rs = pow(0.75 / (PI * rho), 1.0 / 3.0);
    double exchange_potential = -2.0 / (CA_PI_A0 * rs);
    double exchange_epsilon = 0.75 * exchange_potential;
    double ec, vc;
    if (rs >= 1.0) {
        double sqrt_r = sqrt(rs);
        const double g = -0.2846, b1 = 1.0529, b2 = 0.3334;
        ec = g / (1.0 + b1 * sqrt_r + b2 * rs);
        vc = (ec * ec / g) * (1.0 + (7.0 / 6.0) * b1 * sqrt_r + (4.0 / 3.0) * b2 * rs);
    } else {
        double log_r = log(rs);
        const double c1 = 0.0622, c2 = 0.096, c3 = 0.004, c4 = 0.0232, c5 = 0.0192;
        ec = c1 * log_r - c2 + (c3 * log_r - c4) * rs;
        vc = ec - (c1 + (c3 * log_r - c5) * rs) / 3.0;
    }
    *epsilon = exchange_epsilon + ec;
    *potential = exchange_potential + vc;
}

// ca_lda_spin.py for one point with a positive total density.
static __device__ void ca_lsda_point(double rho_up, double rho_down, double* potential_up,
                              double* potential_down, double* energy) {
    const double c1 = 0.0622, c2 = 0.0960, c3 = 0.0040, c4 = 0.0232;
    const double g = -0.2846, b1 = 1.0529, b2 = 0.3334;
    const double as_ = 0.0311, bs = -0.0538, cs = 0.0014, ds = -0.0096;
    const double gs = -0.1686, b1s = 1.3981, b2s = 0.2611;
    double rho = rho_up + rho_down;
    double exchange_up = PW_AX * cbrt(2.0 * rho_up);
    double exchange_down = PW_AX * cbrt(2.0 * rho_down);
    double rs = cbrt(THREE_QUARTERS_OVER_PI / rho);
    double zeta = (rho_up - rho_down) / rho;
    double plus = fmax(1.0 + zeta, 0.0);
    double minus = fmax(1.0 - zeta, 0.0);
    double interpolation = (pow(plus, 4.0 / 3.0) + pow(minus, 4.0 / 3.0) - 2.0) / CA_GAMMA;
    double interpolation_slope = (4.0 / 3.0) * (cbrt(plus) - cbrt(minus)) / CA_GAMMA;
    double unpolarized, unpolarized_potential, polarized, polarized_potential;
    if (rs < 1.0) {
        double log_r = log(rs);
        unpolarized = c1 * log_r - c2 + (c3 * log_r - c4) * rs;
        unpolarized_potential = unpolarized - (c1 + (c3 * log_r + c3 - c4) * rs) / 3.0;
        polarized = as_ * log_r + bs + (cs * log_r + ds) * rs;
        polarized_potential = polarized - (as_ + (cs * log_r + cs + ds) * rs) / 3.0;
    } else {
        double sqrt_r = sqrt(rs);
        unpolarized = g / (1.0 + b1 * sqrt_r + b2 * rs);
        unpolarized_potential = unpolarized * unpolarized
                                * (1.0 + 7.0 * b1 * sqrt_r / 6.0 + 4.0 / 3.0 * b2 * rs) / g;
        polarized = gs / (1.0 + b1s * sqrt_r + b2s * rs);
        polarized_potential = polarized * polarized
                              * (1.0 + 7.0 * b1s * sqrt_r / 6.0 + 4.0 / 3.0 * b2s * rs) / gs;
    }
    double mixed = unpolarized_potential + interpolation * (polarized_potential - unpolarized_potential);
    double difference = polarized - unpolarized;
    double correlation_up = mixed + difference * (1.0 - zeta) * interpolation_slope;
    double correlation_down = mixed - difference * (1.0 + zeta) * interpolation_slope;
    double correlation = unpolarized + interpolation * difference;
    *potential_up = 2.0 * (4.0 / 3.0) * exchange_up + correlation_up;
    *potential_down = 2.0 * (4.0 / 3.0) * exchange_down + correlation_down;
    *energy = 2.0 * (rho_up * exchange_up + rho_down * exchange_down) + correlation * rho;
}

// asymptotic_correction.py _channel_correction at one point.
static __device__ double lb_correction(double rho, double g0, double g1, double g2, double spin_scale,
                                int capped, double cap) {
    if (!(rho > 0.0)) return 0.0;
    double magnitude = sqrt(g0 * g0 + g1 * g1 + g2 * g2);
    double rho13 = cbrt(rho);
    double x = spin_scale * magnitude / (rho * rho13);
    double value = LB_TWO_BETA * (rho13 / spin_scale) * x * x / (1.0 + LB_THREE_BETA * x * asinh(x));
    if (capped) value = fmin(value, cap);
    return value;
}

#ifdef KERNEL_lb_unpolarized
extern "C" __global__ void lb_unpolarized(index_t count, const double* density, const double* gradient,
                                          double spin_scale, int capped, double cap,
                                          double* potential, double* energy_density,
                                          double* epsilon) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double rho = density[w];
    double v = 0.0, eps = 0.0;
    if (rho > 0.0) ca_lda_point(rho, &v, &eps);
    double correction = lb_correction(rho, gradient[w], gradient[count + w], gradient[2 * count + w],
                                      spin_scale, capped, cap);
    potential[w] = v - correction;
    energy_density[w] = rho * eps;
    epsilon[w] = eps;
}
#endif

#ifdef KERNEL_lb_spin
extern "C" __global__ void lb_spin(index_t count, const double* up, const double* down,
                                   const double* gradient_up, const double* gradient_down, int capped,
                                   double cap, double* potential_up, double* potential_down,
                                   double* energy_density) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double rho_up = up[w], rho_down = down[w];
    double vu = 0.0, vd = 0.0, e = 0.0;
    if (rho_up + rho_down > 0.0) ca_lsda_point(rho_up, rho_down, &vu, &vd, &e);
    double cu = lb_correction(rho_up, gradient_up[w], gradient_up[count + w], gradient_up[2 * count + w],
                              1.0, capped, cap);
    double cd = lb_correction(rho_down, gradient_down[w], gradient_down[count + w],
                              gradient_down[2 * count + w], 1.0, capped, cap);
    potential_up[w] = vu - cu;
    potential_down[w] = vd - cd;
    energy_density[w] = e;
}
#endif
