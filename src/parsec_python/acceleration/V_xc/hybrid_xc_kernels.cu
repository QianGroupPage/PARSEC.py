// Semilocal parts of the hybrid functionals on a CUDA device, in float64.
//
// Appended to device_xc_kernels.cu (its constants, stencils, pw92_g, pw92_lsd,
// pbe_exchange, pw91_exchange and pw91_gradient_correction are used here) and
// compiled with --fmad=false.  Every function is a line-by-line translation of
// the readable NumPy code it names (parsec_python.V_xc): becke_lyp.py, ityh.py,
// wpbe.py, semilocal_components.py, pbe.py, pbe_spin.py, pw91.py, pw92_lda.py.
//
// The semilocal part is a weighted sum of the components below, numbered as
// device_hybrid_xc.COMPONENTS; a weight of zero skips a component.  The spin
// kernel works in libxc's variables s_ab = D n_a . D n_b (pbe.py
// discrete_spin_gga_ab): the flux of channel a is 2 vsigma_aa D n_a + vsigma_ab D n_b.

// V_xc/b3lyp.py and becke_lyp.py
#define B88_BETA @B88_BETA@
#define B88_CS @B88_CS@
#define SLATER_CX @SLATER_CX@
#define LYP_A @LYP_A@
#define LYP_B @LYP_B@
#define LYP_C @LYP_C@
#define LYP_D @LYP_D@
#define LYP_CF @LYP_CF@
#define LYP_CF_SPIN @LYP_CF_SPIN@
#define VWN_F_SECOND @VWN_F_SECOND@
#define FZ_DENOMINATOR @FZ_DENOMINATOR@
// V_xc/ityh.py
#define ITYH_X @ITYH_X@
#define ITYH_SERIES_CUTOFF @ITYH_SERIES_CUTOFF@
// V_xc/wpbe.py
#define WP_A @WP_A@
#define WP_B @WP_B@
#define WP_C @WP_C@
#define WP_D @WP_D@
#define WP_E @WP_E@
#define WP_HA1 @WP_HA1@
#define WP_HA2 @WP_HA2@
#define WP_HA3 @WP_HA3@
#define WP_HA4 @WP_HA4@
#define WP_HA5 @WP_HA5@
#define WP_FC1 @WP_FC1@
#define WP_FC2 @WP_FC2@
#define WP_EGA1 @WP_EGA1@
#define WP_EGA2 @WP_EGA2@
#define WP_EGA3 @WP_EGA3@
#define WP_EG_CUTOFF @WP_EG_CUTOFF@
#define WP_NU_CUTOFF @WP_NU_CUTOFF@
#define WP_EB1 @WP_EB1@
#define WP_EA1 @WP_EA1@
#define WP_EA2 @WP_EA2@
#define WP_EA3 @WP_EA3@
#define WP_EA4 @WP_EA4@
#define WP_EA5 @WP_EA5@
#define WP_EA6 @WP_EA6@
#define WP_EA7 @WP_EA7@
#define WP_EA8 @WP_EA8@
#define WP_S_TRANSITION @WP_S_TRANSITION@
#define WP_S_MAX @WP_S_MAX@
#define WP_S_CONST @WP_S_CONST@
#define WP_S_FLOOR @WP_S_FLOOR@
#define WP_S_DERIVATIVE_FLOOR @WP_S_DERIVATIVE_FLOOR@
#define WP_SQRT_PI @WP_SQRT_PI@
#define WP_SQRT_A @WP_SQRT_A@
#define WP_A15 @WP_A15@
#define WP_A25 @WP_A25@
#define WP_A2 @WP_A2@
#define WP_A3 @WP_A3@
#define WP_EULER @WP_EULER@
#define WP_E1_TERMS @WP_E1_TERMS@
#define WP_E1_DEPTH @WP_E1_DEPTH@

#define NCOMPONENTS 12
// becke_lyp.py VWN parameter sets (A, x0, b, c), as in b3lyp.py.
__constant__ double VWN_SETS[5][4] = @VWN_SETS@;
// ityh.py _SERIES (c_1 .. c_8 of F = sum c_k a**(-2k)).
__constant__ double ITYH_SERIES[8] = @ITYH_SERIES@;

// ---------------------------------------------------------------------------
// wpbe.py: scaled E1 and forward-mode derivatives (value, d/ds, d/dnu)
// ---------------------------------------------------------------------------

static __device__ double hx_scaled_exp1(double x) {
    if (x <= 1.0) {
        double total = 0.0, term = 1.0;
        for (int k = 1; k <= WP_E1_TERMS; ++k) {
            term = term * (-x) / k;
            total = total + term / k;
        }
        return exp(x) * (-WP_EULER - log(x) - total);
    }
    double f = x + (2 * WP_E1_DEPTH + 1);
    for (int k = WP_E1_DEPTH; k > 0; --k) f = x + (2 * k - 1) - (double)(k * k) / f;
    return 1.0 / f;
}

struct Dual {
    double v, s, w;
};
static __device__ __forceinline__ Dual dual(double v, double s, double w) { Dual d; d.v = v; d.s = s; d.w = w; return d; }
static __device__ __forceinline__ Dual operator+(Dual a, Dual b) { return dual(a.v + b.v, a.s + b.s, a.w + b.w); }
static __device__ __forceinline__ Dual operator+(Dual a, double b) { return dual(a.v + b, a.s, a.w); }
static __device__ __forceinline__ Dual operator+(double b, Dual a) { return dual(a.v + b, a.s, a.w); }
static __device__ __forceinline__ Dual operator-(Dual a, Dual b) { return dual(a.v - b.v, a.s - b.s, a.w - b.w); }
static __device__ __forceinline__ Dual operator-(Dual a, double b) { return dual(a.v - b, a.s, a.w); }
static __device__ __forceinline__ Dual operator-(double b, Dual a) { return dual(b - a.v, -a.s, -a.w); }
static __device__ __forceinline__ Dual operator-(Dual a) { return dual(-a.v, -a.s, -a.w); }
static __device__ __forceinline__ Dual operator*(Dual a, Dual b) {
    return dual(a.v * b.v, a.s * b.v + a.v * b.s, a.w * b.v + a.v * b.w);
}
static __device__ __forceinline__ Dual operator*(Dual a, double b) { return dual(a.v * b, a.s * b, a.w * b); }
static __device__ __forceinline__ Dual operator*(double b, Dual a) { return dual(a.v * b, a.s * b, a.w * b); }
static __device__ __forceinline__ Dual operator/(Dual a, Dual b) {
    double q = a.v / b.v;
    return dual(q, (a.s - q * b.s) / b.v, (a.w - q * b.w) / b.v);
}
static __device__ __forceinline__ Dual operator/(Dual a, double b) { return dual(a.v / b, a.s / b, a.w / b); }
static __device__ __forceinline__ Dual operator/(double b, Dual a) { return dual(b, 0.0, 0.0) / a; }
static __device__ __forceinline__ Dual chain(Dual x, double value, double derivative) {
    return dual(value, derivative * x.s, derivative * x.w);
}
static __device__ __forceinline__ Dual dpow(Dual x, double exponent) {
    double value = pow(x.v, exponent);
    return chain(x, value, exponent * value / x.v);
}
static __device__ __forceinline__ Dual dsqrt(Dual x) { double r = sqrt(x.v); return chain(x, r, 0.5 / r); }
static __device__ __forceinline__ Dual dlog(Dual x) { return chain(x, log(x.v), 1.0 / x.v); }
static __device__ __forceinline__ Dual derfcx(Dual x) {
    double value = erfcx(x.v);
    return chain(x, value, 2.0 * x.v * value - 2.0 / WP_SQRT_PI);
}
static __device__ __forceinline__ Dual dscaled_exp1(Dual x) {
    double value = hx_scaled_exp1(x.v);
    return chain(x, value, value - 1.0 / x.v);
}

// wpbe.py _enhancement
static __device__ Dual wpbe_enhancement_branch(Dual s, Dual w, int large_nu, int small_s) {
    Dual s2 = s * s;
    Dual s4 = s2 * s2;
    Dual h = (WP_HA1 * s2 + WP_HA2 * s4) / (1.0 + WP_HA3 * s4 + WP_HA4 * s4 * s + WP_HA5 * s4 * s2);
    Dual f = WP_FC1 * h + WP_FC2;
    double eb1 = large_nu ? 2.0 : WP_EB1;
    Dual w2 = w * w;
    Dual s2h = s2 * h;
    Dual aux1 = WP_D + s2h;
    Dual aux2 = (9.0 / (4.0 * WP_A)) * s2h;
    Dual aux3 = aux1 + w2;
    Dual aux4 = s2h + eb1 * w2;
    Dual aux5 = (9.0 / (4.0 * WP_A)) * aux4;
    Dual aux6 = WP_D + aux4;
    Dual eg;
    if (small_s) {
        eg = WP_EGA1 + WP_EGA2 * s2 + WP_EGA3 * s4;
    } else {
        Dual aux1_72 = dpow(aux1, 3.5);
        Dual ga = WP_SQRT_PI * (15.0 * WP_E + 6.0 * WP_C * (1.0 + f * s2) * aux1 + 4.0 * WP_B * aux1 * aux1
                                + 8.0 * WP_A * aux1 * aux1 * aux1) / (16.0 * aux1_72)
                  - (3.0 * PI / 4.0) * WP_SQRT_A * derfcx(dsqrt(aux2));
        Dual gb = (15.0 * WP_SQRT_PI) * s2 / (16.0 * aux1_72);
        eg = -((3.0 * PI / 4.0) + ga) / gb;
    }
    Dual csf = WP_C * s2 * f;
    Dual s2eg = s2 * eg;
    Dual term2 = (aux1 * aux1 * WP_B + aux1 * WP_C + 2.0 * WP_E + aux1 * csf + 2.0 * s2eg) / (2.0 * aux1 * aux1 * aux1);
    Dual aux3_52 = dpow(aux3, 2.5);
    Dual term3 = -w * (4.0 * aux3 * aux3 * WP_B + 6.0 * aux3 * WP_C + 15.0 * WP_E + 6.0 * aux3 * csf + 15.0 * s2eg)
                 / (8.0 * aux1 * aux3_52);
    Dual w3 = w2 * w;
    Dual term4 = -w3 * (aux3 * WP_C + 5.0 * WP_E + aux3 * csf + 5.0 * s2eg) / (2.0 * aux1 * aux1 * aux3_52);
    Dual w5 = w3 * w2;
    Dual term5 = -w5 * (WP_E + s2eg) / (aux1 * aux1 * aux1 * aux3_52);
    Dual term1;
    if (large_nu) {
        term1 = -0.5 * WP_A * (-dscaled_exp1(aux5) + dlog(aux6) - dlog(aux4));
    } else {
        double a12 = WP_SQRT_A;
        Dual w4 = w2 * w2;
        Dual w6 = w4 * w2;
        Dual w7 = w6 * w;
        Dual w8 = w4 * w4;
        Dual np1 = -1.5 * WP_EA1 * a12 * w + (27.0 * WP_EA3 / (8.0 * a12)) * w3
                   - (243.0 * WP_EA5 / (32.0 * WP_A15)) * w5 + (2187.0 * WP_EA7 / (128.0 * WP_A25)) * w7;
        Dual np2 = -WP_A + (9.0 * WP_EA2 / 4.0) * w2 - (81.0 * WP_EA4 / (16.0 * WP_A)) * w4
                   + (729.0 * WP_EA6 / (64.0 * WP_A2)) * w6
                   - (6561.0 * WP_EA8 / (256.0 * WP_A3)) * w8;
        Dual t1 = 0.5 * (np1 * PI * derfcx(dsqrt(aux5)) - np2 * dscaled_exp1(aux5));
        Dual sqrt_aux4 = dsqrt(aux4);
        Dual sqrt_aux6 = dsqrt(aux6);
        Dual aux4_32 = aux4 * sqrt_aux4;
        Dual aux4_52 = aux4_32 * aux4;
        Dual aux6_32 = aux6 * sqrt_aux6;
        Dual aux6_52 = aux6_32 * aux6;
        Dual aux6_72 = aux6_52 * aux6;
        Dual aux6_2 = aux6 * aux6;
        Dual aux6_4 = aux6_2 * aux6_2;
        Dual aux4_2 = aux4 * aux4;
        Dual f2 = (0.5 * WP_EA1 * WP_SQRT_PI * WP_A) / sqrt_aux6;
        Dual f3 = (0.5 * WP_EA2 * WP_A) / aux6;
        Dual f4 = WP_EA3 * WP_SQRT_PI * (-9.0 / (8.0 * sqrt_aux4) + 0.25 * WP_A / aux6_32);
        Dual f5 = (WP_EA4 / 128.0) * (-144.0 / aux4 + 64.0 * WP_A / aux6_2);
        Dual f6 = WP_EA5 * (3.0 * WP_SQRT_PI * (3.0 * aux6_52 * (9.0 * aux4 - 2.0 * WP_A) + 4.0 * aux4_32 * WP_A * WP_A))
                  / (32.0 * aux6_52 * aux4_32 * WP_A);
        Dual f7 = WP_EA6 * (32.0 * WP_A / (aux6_2 * aux6) + (-36.0 + (81.0 / WP_A) * s2h) / aux4_2) / 32.0;
        Dual f8 = WP_EA7 * (-3.0 * WP_SQRT_PI * (-40.0 * aux4_52 * WP_A3
                                               + 9.0 * aux6_72 * (27.0 * aux4_2 - 6.0 * aux4 * WP_A + 4.0 * WP_A * WP_A)))
                  / (128.0 * aux6_72 * aux4_52 * WP_A * WP_A);
        Dual f9 = (324.0 * WP_EA6 * eb1 * aux6_4 * aux4 * WP_A
                   + WP_EA8 * (384.0 * aux4_2 * aux4 * WP_A3 + aux6_4 * (-729.0 * aux4_2 + 324.0 * aux4 * WP_A - 288.0 * WP_A * WP_A)))
                  / (128.0 * aux6_4 * aux4_2 * aux4 * WP_A * WP_A);
        Dual t2t9 = f2 * w + f3 * w2 + f4 * w3 + f5 * w4 + f6 * w5 + f7 * w6 + f8 * w7 + f9 * w8;
        Dual t10 = 0.5 * WP_A * dlog(aux4 / aux6);
        term1 = t1 + t2t9 + t10;
    }
    return (-8.0 / 9.0) * (term1 + term2 + term3 + term4 + term5);
}

// wpbe.py wpbe_sr_exchange_partials for one positive density (Heyd's rescaling of s).
static __device__ void wpbe_sr_exchange(double n, double sigma, double omega, double* e, double* e_n, double* e_sigma) {
    double n13 = cbrt(n);
    double kf = KF * n13;
    double s = sqrt(sigma) / (2.0 * kf * n);
    double nu = omega / kf;
    double effective, slope;
    if (s > WP_S_TRANSITION) {
        effective = WP_S_MAX - WP_S_CONST / (s * s);
        slope = 2.0 * WP_S_CONST / (s * s * s);
    } else {
        effective = s;
        slope = 1.0;
    }
    if (effective < WP_S_FLOOR) {
        effective = WP_S_FLOOR;
        slope = 0.0;
    }
    int large_nu = nu > WP_NU_CUTOFF;
    Dual result = wpbe_enhancement_branch(dual(effective, 1.0, 0.0), dual(nu, 0.0, 1.0), large_nu,
                                          effective <= WP_EG_CUTOFF);
    double value = result.v;
    double d_s = result.s * slope;
    double d_nu = result.w;
    double d_s_over_s;
    if (s < WP_S_DERIVATIVE_FLOOR) {
        Dual floor_result = wpbe_enhancement_branch(dual(WP_S_DERIVATIVE_FLOOR, 1.0, 0.0), dual(nu, 0.0, 1.0),
                                                    large_nu, 1);
        d_s_over_s = floor_result.s / WP_S_DERIVATIVE_FLOOR;
    } else {
        d_s_over_s = d_s / s;
    }
    double prefactor = -SLATER_CX * n * n13;
    *e = prefactor * value;
    *e_n = (4.0 / 3.0) * (-SLATER_CX * n13) * value
           + prefactor * (d_s * (-(4.0 / 3.0) * s / n) + d_nu * (-nu / (3.0 * n)));
    *e_sigma = prefactor * d_s_over_s / (8.0 * kf * kf * n * n);
}

// ---------------------------------------------------------------------------
// becke_lyp.py: Slater, B88, VWN, LYP; ityh.py
// ---------------------------------------------------------------------------

static __device__ void slater_exchange(double n, double* e, double* e_n, double* e_sigma) {
    double n13 = cbrt(n);
    *e = -SLATER_CX * n * n13;
    *e_n = -(4.0 / 3.0) * SLATER_CX * n13;
    *e_sigma = 0.0;
}

static __device__ void b88_spin_channel(double rho, double sigma, double* e, double* e_rho, double* e_sigma) {
    double rho13 = cbrt(rho);
    double rho43 = rho * rho13;
    double x = sqrt(sigma) / rho43;
    double as = asinh(x);
    double denominator = 1.0 + 6.0 * B88_BETA * x * as;
    double g = x * x / denominator;
    double denominator_prime = 6.0 * B88_BETA * (as + x / sqrt(1.0 + x * x));
    double g_prime_over_x = (2.0 * denominator - x * denominator_prime) / (denominator * denominator);
    *e = -rho43 * (B88_CS + B88_BETA * g);
    *e_rho = -(4.0 / 3.0) * rho13 * (B88_CS + B88_BETA * (g - x * x * g_prime_over_x));
    *e_sigma = -B88_BETA * g_prime_over_x / (2.0 * rho43);
}

static __device__ void b88_exchange(double n, double sigma, double* e, double* e_n, double* e_sigma) {
    double energy, d_rho, d_sigma;
    b88_spin_channel(0.5 * n, 0.25 * sigma, &energy, &d_rho, &d_sigma);
    *e = 2.0 * energy;
    *e_n = d_rho;
    *e_sigma = 0.5 * d_sigma;
}

// ityh.py erf_attenuation
static __device__ void erf_attenuation(double a, double* value, double* derivative) {
    if (a < ITYH_SERIES_CUTOFF) {
        double e_minus_one = expm1(-1.0 / (4.0 * a * a));
        double bracket = WP_SQRT_PI * erf(1.0 / (2.0 * a)) + 2.0 * a * (e_minus_one - (2.0 * a * a * e_minus_one + 0.5));
        *value = 1.0 - (8.0 / 3.0) * a * bracket;
        *derivative = -(8.0 / 3.0) * (bracket - 12.0 * a * a * a * e_minus_one - 3.0 * a);
    } else {
        double inverse2 = 1.0 / (a * a);
        double total = 0.0, slope = 0.0;
        for (int k = 8; k > 0; --k) {
            total = (total + ITYH_SERIES[k - 1]) * inverse2;
            slope = (slope + k * ITYH_SERIES[k - 1]) * inverse2;
        }
        *value = total;
        *derivative = -2.0 * slope / a;
    }
}

static __device__ void ityh_spin_channel(double rho, double sigma, double omega, double* e, double* e_rho, double* e_sigma) {
    double energy, d_rho, d_sigma;
    b88_spin_channel(rho, sigma, &energy, &d_rho, &d_sigma);
    double rho13 = cbrt(rho);
    double lda = -ITYH_X * rho * rho13;
    double enhancement = energy / lda;
    double d_enhancement_rho = (d_rho - enhancement * (4.0 / 3.0) * (-ITYH_X * rho13)) / lda;
    double d_enhancement_sigma = d_sigma / lda;
    double a = 0.5 * omega * sqrt(2.0 * ITYH_X * enhancement / (9.0 * PI)) / rho13;
    double attenuation, slope;
    erf_attenuation(a, &attenuation, &slope);
    double a_rho = a * (-1.0 / (3.0 * rho) + 0.5 * d_enhancement_rho / enhancement);
    double a_sigma = a * 0.5 * d_enhancement_sigma / enhancement;
    *e = energy * attenuation;
    *e_rho = d_rho * attenuation + energy * slope * a_rho;
    *e_sigma = d_sigma * attenuation + energy * slope * a_sigma;
}

static __device__ void ityh_exchange(double n, double sigma, double omega, double* e, double* e_n, double* e_sigma) {
    double energy, d_rho, d_sigma;
    ityh_spin_channel(0.5 * n, 0.25 * sigma, omega, &energy, &d_rho, &d_sigma);
    *e = 2.0 * energy;
    *e_n = d_rho;
    *e_sigma = 0.5 * d_sigma;
}

// becke_lyp.py _vwn_epsilon_partials for the set ``index`` of VWN_SETS
// (0 rpa paramagnetic, 1 vwn5 paramagnetic, 2 rpa ferromagnetic, 3 vwn5 ferromagnetic, 4 spin stiffness).
static __device__ void vwn_epsilon(double x, int index, double* epsilon, double* derivative) {
    double a = VWN_SETS[index][0], x0 = VWN_SETS[index][1], b = VWN_SETS[index][2], c = VWN_SETS[index][3];
    double big_x = x * x + b * x + c;
    double big_x0 = x0 * x0 + b * x0 + c;
    double q = sqrt(4.0 * c - b * b);
    double two_x_b = 2.0 * x + b;
    double arctangent = atan(q / two_x_b);
    double ratio = b * x0 / big_x0;
    *epsilon = a * (log(x * x / big_x) + (2.0 * b / q) * arctangent
                    - ratio * (log((x - x0) * (x - x0) / big_x) + (2.0 * (b + 2.0 * x0) / q) * arctangent));
    double four_x = 4.0 * big_x;
    *derivative = a * (2.0 / x - two_x_b / big_x - 4.0 * b / four_x
                       - ratio * (2.0 / (x - x0) - two_x_b / big_x - 4.0 * (b + 2.0 * x0) / four_x));
}

static __device__ double vwn_x(double n) { return sqrt(cbrt(3.0 / (4.0 * PI * n))); }

static __device__ void vwn_correlation(double n, int vwn5, double* e, double* e_n) {
    double x = vwn_x(n);
    double epsilon, epsilon_x;
    vwn_epsilon(x, vwn5 ? 1 : 0, &epsilon, &epsilon_x);
    *e = n * epsilon;
    *e_n = epsilon - x * epsilon_x / 6.0;
}

static __device__ void vwn_spin_correlation(double n, double up, double down, int vwn5, double* e, double* e_up,
                                            double* e_down) {
    double x = vwn_x(n);
    double zeta = (up - down) / n;
    double plus = fmax(1.0 + zeta, 0.0);
    double minus = fmax(1.0 - zeta, 0.0);
    double f = (pow(plus, 4.0 / 3.0) + pow(minus, 4.0 / 3.0) - 2.0) / FZ_DENOMINATOR;
    double f_zeta = (4.0 / 3.0) * (cbrt(plus) - cbrt(minus)) / FZ_DENOMINATOR;
    double paramagnetic, paramagnetic_x, ferromagnetic, ferromagnetic_x;
    vwn_epsilon(x, vwn5 ? 1 : 0, &paramagnetic, &paramagnetic_x);
    vwn_epsilon(x, vwn5 ? 3 : 2, &ferromagnetic, &ferromagnetic_x);
    double difference = ferromagnetic - paramagnetic;
    double difference_x = ferromagnetic_x - paramagnetic_x;
    double epsilon, epsilon_x, epsilon_zeta;
    if (!vwn5) {
        epsilon = paramagnetic + difference * f;
        epsilon_x = paramagnetic_x + difference_x * f;
        epsilon_zeta = difference * f_zeta;
    } else {
        double stiffness, stiffness_x;
        vwn_epsilon(x, 4, &stiffness, &stiffness_x);
        double z3 = pow(zeta, 3.0);
        double z4 = zeta * z3;
        epsilon = paramagnetic + stiffness * f * (1.0 - z4) / VWN_F_SECOND + difference * f * z4;
        epsilon_x = paramagnetic_x + stiffness_x * f * (1.0 - z4) / VWN_F_SECOND + difference_x * f * z4;
        epsilon_zeta = stiffness * (f_zeta * (1.0 - z4) - 4.0 * z3 * f) / VWN_F_SECOND
                       + difference * (f_zeta * z4 + 4.0 * z3 * f);
    }
    double common = epsilon - x * epsilon_x / 6.0;
    *e = n * epsilon;
    *e_up = common + (1.0 - zeta) * epsilon_zeta;
    *e_down = common - (1.0 + zeta) * epsilon_zeta;
}

static __device__ void lyp_correlation(double n, double sigma, double* e, double* e_n, double* e_sigma) {
    double u = 1.0 / cbrt(n);
    double denominator = 1.0 + LYP_D * u;
    double delta = LYP_C * u + LYP_D * u / denominator;
    double du = LYP_D * u / denominator;
    double delta_n = -(delta - du * du) / (3.0 * n);
    double p = exp(-LYP_C * u) / denominator;
    double p_n = p * delta / (3.0 * n);
    double u5 = pow(u, 5.0);
    double k = 1.0 / 24.0 + 7.0 * delta / 72.0;
    double bracket = LYP_CF * n - u5 * sigma * k;
    double bracket_n = LYP_CF - sigma * (-(5.0 / 3.0) * u5 * k / n + u5 * (7.0 / 72.0) * delta_n);
    double ab = LYP_A * LYP_B;
    *e = -LYP_A * n / denominator - ab * p * bracket;
    *e_n = -LYP_A * (1.0 / denominator + LYP_D * u / (3.0 * denominator * denominator))
           - ab * (p_n * bracket + p * bracket_n);
    *e_sigma = ab * p * u5 * k;
}

// out: e, e_a, e_b, e_saa, e_sab, e_sbb
static __device__ void lyp_spin_correlation(double n, double a, double b, double s_aa, double s_ab, double s_bb,
                                            double* out) {
    double u = 1.0 / cbrt(n);
    double denominator = 1.0 + LYP_D * u;
    double delta = LYP_C * u + LYP_D * u / denominator;
    double du = LYP_D * u / denominator;
    double delta_n = -(delta - du * du) / (3.0 * n);
    double omega_lyp = exp(-LYP_C * u) / denominator * pow(u, 11.0);
    double omega_n = -omega_lyp * (11.0 - delta) / (3.0 * n);
    double ab = a * b;
    double cf = LYP_CF_SPIN;
    double weighted = (a * s_aa + b * s_bb) / n;
    double same = s_aa + s_bb;
    double mixed = (47.0 - 7.0 * delta) / 9.0;
    double diagonal = 1.0 / 9.0 - delta / 3.0;
    double eleven = (delta - 11.0) / 9.0;
    double bracket = cf * (pow(a, 8.0 / 3.0) + pow(b, 8.0 / 3.0)) + mixed * s_ab + diagonal * same - eleven * weighted;
    double n2 = n * n;
    double g = -(4.0 / 3.0) * n2 * s_ab - a * a * s_bb - b * b * s_aa;
    double w = ab * bracket + g;
    double common_bracket = delta_n * (-(7.0 / 9.0) * s_ab - same / 3.0 - weighted / 9.0) + eleven * weighted / n;
    double bracket_a = (8.0 / 3.0) * cf * pow(a, 5.0 / 3.0) + common_bracket - eleven * s_aa / n;
    double bracket_b = (8.0 / 3.0) * cf * pow(b, 5.0 / 3.0) + common_bracket - eleven * s_bb / n;
    double g_a = -(8.0 / 3.0) * n * s_ab - 2.0 * a * s_bb;
    double g_b = -(8.0 / 3.0) * n * s_ab - 2.0 * b * s_aa;
    double w_a = b * bracket + ab * bracket_a + g_a;
    double w_b = a * bracket + ab * bracket_b + g_b;
    double nd = n * denominator;
    double nd_n = denominator - LYP_D * u / 3.0;
    double first = -4.0 * LYP_A * ab / nd;
    double first_n = 4.0 * LYP_A * ab * nd_n / (nd * nd);
    double prefactor = LYP_A * LYP_B;
    out[0] = first - prefactor * omega_lyp * w;
    out[1] = -4.0 * LYP_A * b / nd + first_n - prefactor * (omega_n * w + omega_lyp * w_a);
    out[2] = -4.0 * LYP_A * a / nd + first_n - prefactor * (omega_n * w + omega_lyp * w_b);
    out[3] = -prefactor * omega_lyp * (ab * (diagonal - eleven * a / n) - b * b);
    out[4] = -prefactor * omega_lyp * (ab * mixed - (4.0 / 3.0) * n2);
    out[5] = -prefactor * omega_lyp * (ab * (diagonal - eleven * b / n) - a * a);
}

// ---------------------------------------------------------------------------
// PBE and PW91 correlation halves (pbe.py _correlation_partials, pbe_spin.py
// _spin_correlation_partials, pw91.py), PW92 with Fortran digits
// ---------------------------------------------------------------------------

static __device__ void pbe_correlation(double n, double sig, double* e, double* e_n, double* e_sigma) {
    double n13 = cbrt(n);
    double kf = KF * n13;
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
    *e = n * (epsilon_lda + h);
    *e_n = epsilon_lda + h + n * (epsilon_lda_n + h_n);
    *e_sigma = n * h_sigma;
}

// out: e, e_up, e_down, e_stot
static __device__ void pbe_spin_correlation(double n, double up, double down, double sigma_total, double* out) {
    double sig = fmax(sigma_total, 0.0);
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
    double a_exponent = -beta_over_gamma * exp(exponent) / (exponential_minus_one * exponential_minus_one);
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
    out[0] = n * per_electron;
    out[1] = common + (1.0 - zeta) * per_electron_zeta;
    out[2] = common - (1.0 + zeta) * per_electron_zeta;
    out[3] = n * h_t * dt2_dsigma;
}

static __device__ void pw91_correlation(double n, double sig, double* e, double* e_n, double* e_sigma) {
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
    *e = n * (ec + h);
    *e_n = ec - rs * ecrs / 3.0 + h + hrs - (7.0 / 6.0) * t2 * ht;
    *e_sigma = 0.5 * n * ht * dt2_dsigma;
}

static __device__ void pw91_spin_correlation(double n, double up, double down, double sigma_total, double* out) {
    double sig = fmax(sigma_total, 0.0);
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
    double gz = (pow(one_plus * one_plus + PW91_ETA, -1.0 / 6.0) - pow(one_minus * one_minus + PW91_ETA, -1.0 / 6.0)) / 3.0;
    double dt2_dsigma = PI / (16.0 * g * g * kf * n * n);
    double t2 = sig * dt2_dsigma;
    double correction[4];
    pw91_gradient_correction(rs, g, gz, ec, ecrs, eczet, t2, correction);
    double h = correction[0], hrs = correction[1], ht = correction[2], hz = correction[3];
    double pref = hz - gz * t2 * ht / g;
    double common = ec - rs * ecrs / 3.0 + h + hrs - (7.0 / 6.0) * t2 * ht;
    out[0] = n * (ec + h);
    out[1] = common + (1.0 - zeta) * (eczet + pref);
    out[2] = common - (1.0 + zeta) * (eczet + pref);
    out[3] = 0.5 * n * ht * dt2_dsigma;
}

static __device__ void pw92_correlation(double n, double* e, double* e_n) {
    double rs = cbrt(3.0 / (4.0 * PI * n));
    double epsilon, epsilon_rs, epsilon_zeta;
    pw92_lsd(rs, 0.0, PW_A_UNPOLARIZED, PW_GAM, PW_FZZ, &epsilon, &epsilon_rs, &epsilon_zeta);
    *e = n * epsilon;
    *e_n = epsilon - rs * epsilon_rs / 3.0;
}

static __device__ void pw92_spin_correlation(double n, double up, double down, double* e, double* e_up, double* e_down) {
    double rs = cbrt(3.0 / (4.0 * PI * n));
    double zeta = (up - down) / n;
    double epsilon, epsilon_rs, epsilon_zeta;
    pw92_lsd(rs, zeta, PW_A_UNPOLARIZED, PW_GAM, PW_FZZ, &epsilon, &epsilon_rs, &epsilon_zeta);
    double common = epsilon - rs * epsilon_rs / 3.0;
    *e = n * epsilon;
    *e_up = common + (1.0 - zeta) * epsilon_zeta;
    *e_down = common - (1.0 + zeta) * epsilon_zeta;
}

// ---------------------------------------------------------------------------
// Weighted sums (semilocal_components.py ComponentSemilocal)
// ---------------------------------------------------------------------------

// Unpolarized exchange core of component c (0..5) at a positive density.
static __device__ void exchange_core(int c, double n, double sigma, double omega, double* e, double* e_n, double* e_s) {
    switch (c) {
        case 0: slater_exchange(n, e, e_n, e_s); break;
        case 1: pbe_exchange(n, sigma, e, e_n, e_s); break;
        case 2: b88_exchange(n, sigma, e, e_n, e_s); break;
        case 3: pw91_exchange(n, sigma, e, e_n, e_s); break;
        case 4: wpbe_sr_exchange(n, sigma, omega, e, e_n, e_s); break;
        default: ityh_exchange(n, sigma, omega, e, e_n, e_s); break;
    }
}

// Unpolarized correlation core of component c (6..11) at a positive density.
static __device__ void correlation_core(int c, double n, double sigma, double* e, double* e_n, double* e_s) {
    *e_s = 0.0;
    switch (c) {
        case 6: pbe_correlation(n, sigma, e, e_n, e_s); break;
        case 7: pw91_correlation(n, sigma, e, e_n, e_s); break;
        case 8: pw92_correlation(n, e, e_n); break;
        case 9: vwn_correlation(n, 0, e, e_n); break;
        case 10: vwn_correlation(n, 1, e, e_n); break;
        default: lyp_correlation(n, sigma, e, e_n, e_s); break;
    }
}

// out: e, vrho, vsigma (Hartree, per volume) of the weighted sum.
static __device__ void hybrid_unpolarized(double rho, double sigma, const double* weights, const double* omegas,
                                          double* out) {
    double n = fmax(rho, 0.0);
    double sig = fmax(sigma, 0.0);
    double energy = 0.0, d_n = 0.0, d_sigma = 0.0;
    if (n > DENSITY_THRESHOLD) {
        for (int c = 0; c < NCOMPONENTS; ++c) {
            double weight = weights[c];
            if (weight == 0.0) continue;
            double e, en, es;
            if (c < 6) exchange_core(c, n, sig, omegas[c], &e, &en, &es);
            else correlation_core(c, n, sig, &e, &en, &es);
            energy += weight * e;
            d_n += weight * en;
            d_sigma += weight * es;
        }
    }
    out[0] = energy;
    out[1] = d_n;
    out[2] = d_sigma;
}

// out: e, vrho_up, vrho_down, vsigma_uu, vsigma_ud, vsigma_dd (libxc's variables).
static __device__ void hybrid_polarized(double rho_up, double rho_down, double s_uu_in, double s_ud, double s_dd_in,
                                        const double* weights, const double* omegas, double* out) {
    double up = fmax(rho_up, 0.0);
    double down = fmax(rho_down, 0.0);
    double s_uu = fmax(s_uu_in, 0.0);
    double s_dd = fmax(s_dd_in, 0.0);
    double total[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double n = up + down;
    for (int c = 0; c < NCOMPONENTS; ++c) {
        double weight = weights[c];
        if (weight == 0.0) continue;
        double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (c < 6) {
            // semilocal_components.py spin_scaled_exchange
            double doubled = 2.0 * up;
            if (doubled > DENSITY_THRESHOLD) {
                double e, en, es;
                exchange_core(c, doubled, fmax(4.0 * s_uu, 0.0), omegas[c], &e, &en, &es);
                v[0] += 0.5 * e;
                v[1] += en;
                v[3] += 2.0 * es;
            }
            doubled = 2.0 * down;
            if (doubled > DENSITY_THRESHOLD) {
                double e, en, es;
                exchange_core(c, doubled, fmax(4.0 * s_dd, 0.0), omegas[c], &e, &en, &es);
                v[0] += 0.5 * e;
                v[2] += en;
                v[5] += 2.0 * es;
            }
        } else if (n > DENSITY_THRESHOLD) {
            if (c == 6 || c == 7) {
                // semilocal_components.py _total_gradient_form
                double s_tot = fmax(s_uu + 2.0 * s_ud + s_dd, 0.0);
                double r[4];
                if (c == 6) pbe_spin_correlation(n, up, down, s_tot, r);
                else pw91_spin_correlation(n, up, down, s_tot, r);
                v[0] = r[0];
                v[1] = r[1];
                v[2] = r[2];
                v[3] = r[3];
                v[4] = 2.0 * r[3];
                v[5] = r[3];
            } else if (c == 8) {
                pw92_spin_correlation(n, up, down, &v[0], &v[1], &v[2]);
            } else if (c == 9 || c == 10) {
                vwn_spin_correlation(n, up, down, c == 10, &v[0], &v[1], &v[2]);
            } else {
                lyp_spin_correlation(n, up, down, s_uu, s_ud, s_dd, v);
            }
        }
        for (int k = 0; k < 6; ++k) total[k] += weight * v[k];
    }
    for (int k = 0; k < 6; ++k) out[k] = total[k];
}

// The grid kernels: as GGA_UNPOLARIZED_KERNEL and GGA_SPIN_KERNEL of device_xc_kernels.cu.
#ifdef KERNEL_hybrid_gga
extern "C" __global__ void hybrid_gga(index_t count, const double* density, const double* gradient,
                                      const double* weights, const double* omegas, double* energy_density,
                                      double* f_n, double* flux, double* epsilon) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double g0 = gradient[w], g1 = gradient[count + w], g2 = gradient[2 * count + w];
    double sigma = 0.0;
    sigma += g0 * g0;
    sigma += g1 * g1;
    sigma += g2 * g2;
    double n = density[w];
    double out[3];
    hybrid_unpolarized(n, sigma, weights, omegas, out);
    double energy = 2.0 * out[0];
    energy_density[w] = energy;
    f_n[w] = out[1];
    flux[w] = 2.0 * out[2] * g0;
    flux[count + w] = 2.0 * out[2] * g1;
    flux[2 * count + w] = 2.0 * out[2] * g2;
    epsilon[w] = n > DENSITY_THRESHOLD ? energy / n : 0.0;
}
#endif

#ifdef KERNEL_hybrid_spin_gga
extern "C" __global__ void hybrid_spin_gga(index_t count, const double* up, const double* down,
                                           const double* gradient_up, const double* gradient_down,
                                           const double* weights, const double* omegas, double* energy_density,
                                           double* f_up, double* f_down, double* flux_up, double* flux_down) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    double gu[3], gd[3];
    for (int axis = 0; axis < 3; ++axis) {
        gu[axis] = gradient_up[axis * count + w];
        gd[axis] = gradient_down[axis * count + w];
    }
    // pbe.py discrete_spin_gga_ab: sum(g * g for g in grad) from zero.
    double s_uu = 0.0, s_ud = 0.0, s_dd = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        s_uu += gu[axis] * gu[axis];
        s_ud += gu[axis] * gd[axis];
        s_dd += gd[axis] * gd[axis];
    }
    double out[6];
    hybrid_polarized(up[w], down[w], s_uu, s_ud, s_dd, weights, omegas, out);
    energy_density[w] = 2.0 * out[0];
    f_up[w] = out[1];
    f_down[w] = out[2];
    for (int axis = 0; axis < 3; ++axis) {
        flux_up[axis * count + w] = 2.0 * out[3] * gu[axis] + out[4] * gd[axis];
        flux_down[axis * count + w] = 2.0 * out[5] * gd[axis] + out[4] * gu[axis];
    }
}
#endif

// Pointwise test kernel: the partials at given (rho, sigma) or spin variables.
#ifdef KERNEL_hybrid_points
extern "C" __global__ void hybrid_points(index_t count, int polarized, const double* inputs, const double* weights,
                                         const double* omegas, double* outputs) {
    index_t w = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= count) return;
    if (polarized) {
        double out[6];
        hybrid_polarized(inputs[w], inputs[count + w], inputs[2 * count + w], inputs[3 * count + w],
                         inputs[4 * count + w], weights, omegas, out);
        for (int k = 0; k < 6; ++k) outputs[k * count + w] = out[k];
    } else {
        double out[3];
        hybrid_unpolarized(inputs[w], inputs[count + w], weights, omegas, out);
        for (int k = 0; k < 3; ++k) outputs[k * count + w] = out[k];
    }
}
#endif
