#include "hartree_boundary.h"
#include "openmp_workload.h"

#include <pybind11/complex.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace parsec_accelerated_native {
namespace {

// The normalization prefactor becomes subnormal at l = m = 86.
constexpr int kMaximumMultipoleOrder = 60;
// The stride of the angular arrays of extensions before 0.6.0, whose largest
// order was 9.  The GPU kernels of the Python trees of that time read the
// exported prefactors at l*10+m whatever the order.
constexpr int kFormerAngularStride = 10;
constexpr std::size_t kMomentBlockSize = 4096;
constexpr std::size_t kMaximumSymmetryCoefficientBytes =
    512ULL * 1024ULL * 1024ULL;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct AngularGeometry {
    double radius;
    double cosine;
    double sine;
    double phase_real;
    double phase_imag;
};

AngularGeometry angular_geometry(double x, double y, double z) {
    const double radius = std::sqrt(x * x + y * y + z * z);
    double cosine = 1.0;
    if (radius > 0.0) {
        cosine = std::clamp(z / radius, -1.0, 1.0);
    }
    const double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
    const double xy_radius = std::hypot(x, y);
    double phase_real = 1.0;
    double phase_imag = 0.0;
    if (xy_radius > 0.0) {
        phase_real = x / xy_radius;
        phase_imag = y / xy_radius;
    }
    return {radius, cosine, sine, phase_real, phase_imag};
}

double factorial(int value) {
    double result = 1.0;
    for (int integer = 2; integer <= value; ++integer) {
        result *= static_cast<double>(integer);
    }
    return result;
}

std::vector<double> second_derivative_coefficients(int expansion_order) {
    if (
        expansion_order < 2 || expansion_order > 20 ||
        expansion_order % 2 != 0
    ) {
        throw std::invalid_argument(
            "expansion_order must be an even integer from 2 to 20"
        );
    }
    const int width = expansion_order / 2;
    std::vector<double> coefficients(2 * width + 1, 0.0);
    double positive_sum = 0.0;
    for (int shell = 1; shell <= width; ++shell) {
        const double sign = shell % 2 == 0 ? -1.0 : 1.0;
        const double coefficient =
            2.0 * sign * factorial(width) * factorial(width) /
            (
                static_cast<double>(shell * shell) *
                factorial(width - shell) * factorial(width + shell)
            );
        coefficients[width - shell] = coefficient;
        coefficients[width + shell] = coefficient;
        positive_sum += coefficient;
    }
    coefficients[width] = -2.0 * positive_sum;
    return coefficients;
}

// Arrays over (l, m) are stored at l*stride+m with stride = order+1.
std::size_t angular_index(int stride, int angular_momentum, int magnetic) {
    return static_cast<std::size_t>(angular_momentum * stride + magnetic);
}

std::size_t compact_angular_index(int angular_momentum, int magnetic) {
    return static_cast<std::size_t>(
        angular_momentum * (angular_momentum + 1) / 2 + magnetic
    );
}

void evaluate_positive_harmonics(
    double radius,
    double cosine,
    double sine,
    double phase_real,
    double phase_imag,
    int order,
    const std::vector<double>& normalization,
    std::vector<std::complex<double>>& harmonic,
    std::vector<double>& radius_power
) {
    const int stride = order + 1;
    const std::complex<double> phase_unit(phase_real, phase_imag);
    std::complex<double> phase(1.0, 0.0);
    double diagonal = 1.0;  // P_0^0, including the Condon--Shortley phase.

    radius_power[0] = 1.0;
    for (int angular_momentum = 1; angular_momentum <= order;
         ++angular_momentum) {
        radius_power[angular_momentum] =
            radius_power[angular_momentum - 1] * radius;
    }

    for (int magnetic = 0; magnetic <= order; ++magnetic) {
        if (magnetic > 0) {
            diagonal *= -static_cast<double>(2 * magnetic - 1) * sine;
            phase *= phase_unit;
        }

        double previous = diagonal;
        harmonic[angular_index(stride, magnetic, magnetic)] =
            normalization[angular_index(stride, magnetic, magnetic)] *
            previous * phase;
        if (magnetic == order) {
            continue;
        }

        double current =
            static_cast<double>(2 * magnetic + 1) * cosine * diagonal;
        harmonic[angular_index(stride, magnetic + 1, magnetic)] =
            normalization[angular_index(stride, magnetic + 1, magnetic)] *
            current * phase;
        for (int angular_momentum = magnetic + 2;
             angular_momentum <= order; ++angular_momentum) {
            const double following =
                (
                    static_cast<double>(2 * angular_momentum - 1) *
                        cosine * current -
                    static_cast<double>(angular_momentum + magnetic - 1) *
                        previous
                ) /
                static_cast<double>(angular_momentum - magnetic);
            harmonic[angular_index(stride, angular_momentum, magnetic)] =
                normalization[angular_index(stride, angular_momentum, magnetic)] *
                following * phase;
            previous = current;
            current = following;
        }
    }
}

double boundary_potential(
    double radius,
    double cosine,
    double sine,
    double phase_real,
    double phase_imag,
    int order,
    const std::vector<double>& normalization,
    const std::vector<std::complex<double>>& moments,
    std::vector<std::complex<double>>& harmonic,
    std::vector<double>& unused_radius_power
) {
    if (!(radius > 0.0)) {
        throw std::runtime_error(
            "multipole boundary potential is undefined at the origin"
        );
    }
    const int stride = order + 1;
    evaluate_positive_harmonics(
        radius,
        cosine,
        sine,
        phase_real,
        phase_imag,
        order,
        normalization,
        harmonic,
        unused_radius_power
    );

    double result = 0.0;
    double inverse_radius_power = 1.0 / radius;
    for (int angular_momentum = 0; angular_momentum <= order;
         ++angular_momentum) {
        const double factor =
            4.0 * kPi / static_cast<double>(2 * angular_momentum + 1) *
            inverse_radius_power;
        for (int magnetic = 0; magnetic <= angular_momentum; ++magnetic) {
            const std::size_t index =
                angular_index(stride, angular_momentum, magnetic);
            const double positive_real =
                std::real(moments[index] * harmonic[index]);
            // For a real density, Q_l,-m Y_l,-m is the conjugate of the
            // positive-m product.  The pair therefore contributes 2*Re(...).
            result += factor * (magnetic == 0 ? positive_real
                                              : 2.0 * positive_real);
        }
        inverse_radius_power /= radius;
    }
    // PARSEC uses Rydberg electrostatic units: V_H = 2*integral(rho/r) dr.
    return 2.0 * result;
}

}  // namespace

MultipoleBoundaryBuilder::MultipoleBoundaryBuilder(
    int multipole_order,
    double volume_element,
    const IndexArray& multiplicities,
    const py::array_t<
        std::complex<double>,
        py::array::c_style | py::array::forcecast
    >& moment_coefficients,
    const IndexArray& boundary_indptr,
    const FloatArray& boundary_operator_coefficient,
    const FloatArray& boundary_radius,
    const FloatArray& boundary_cosine,
    const FloatArray& boundary_sine,
    const FloatArray& boundary_phase_real,
    const FloatArray& boundary_phase_imag
) {
    if (multipole_order < 0 || multipole_order > kMaximumMultipoleOrder) {
        throw std::invalid_argument("multipole_order must be between 0 and 60");
    }
    if (!(volume_element > 0.0) || !std::isfinite(volume_element)) {
        throw std::invalid_argument("volume_element must be finite and positive");
    }
    if (multiplicities.ndim() != 1 || multiplicities.shape(0) < 1) {
        throw std::invalid_argument("multiplicities must be a nonempty vector");
    }
    const std::size_t wedge_count = static_cast<std::size_t>(
        multiplicities.shape(0)
    );
    const std::size_t angular_count = static_cast<std::size_t>(
        (multipole_order + 1) * (multipole_order + 2) / 2
    );
    if (
        moment_coefficients.ndim() != 2 ||
        static_cast<std::size_t>(moment_coefficients.shape(0)) != wedge_count ||
        static_cast<std::size_t>(moment_coefficients.shape(1)) != angular_count
    ) {
        throw std::invalid_argument(
            "moment_coefficients must have shape (wedge_size, angular_count)"
        );
    }
    if (
        boundary_indptr.ndim() != 1 ||
        static_cast<std::size_t>(boundary_indptr.shape(0)) != wedge_count + 1
    ) {
        throw std::invalid_argument(
            "boundary_indptr must have shape (wedge_size + 1,)"
        );
    }
    const auto indptr = boundary_indptr.unchecked<1>();
    if (indptr(0) != 0) {
        throw std::invalid_argument("boundary_indptr must start at zero");
    }
    for (std::size_t row = 0; row < wedge_count; ++row) {
        if (indptr(static_cast<py::ssize_t>(row + 1)) <
            indptr(static_cast<py::ssize_t>(row))) {
            throw std::invalid_argument(
                "boundary_indptr must be monotonically nondecreasing"
            );
        }
    }
    const std::int64_t signed_term_count = indptr(
        static_cast<py::ssize_t>(wedge_count)
    );
    if (signed_term_count < 0) {
        throw std::invalid_argument("boundary_indptr contains a negative count");
    }
    const std::size_t term_count = static_cast<std::size_t>(signed_term_count);
    const auto check_terms = [term_count](
        const FloatArray& values, const char* name
    ) {
        if (
            values.ndim() != 1 ||
            static_cast<std::size_t>(values.shape(0)) != term_count
        ) {
            throw std::invalid_argument(
                std::string(name) + " must match boundary_indptr"
            );
        }
        for (std::size_t term = 0; term < term_count; ++term) {
            if (!std::isfinite(values.data()[term])) {
                throw std::invalid_argument(
                    std::string(name) + " must contain finite values"
                );
            }
        }
    };
    check_terms(boundary_operator_coefficient, "boundary_operator_coefficient");
    check_terms(boundary_radius, "boundary_radius");
    check_terms(boundary_cosine, "boundary_cosine");
    check_terms(boundary_sine, "boundary_sine");
    check_terms(boundary_phase_real, "boundary_phase_real");
    check_terms(boundary_phase_imag, "boundary_phase_imag");

    point_count_ = wedge_count;
    multipole_order_ = multipole_order;
    stride_ = multipole_order + 1;
    storage_ = static_cast<std::size_t>(stride_) * static_cast<std::size_t>(stride_);
    volume_element_ = volume_element;
    symmetry_angular_count_ = angular_count;
    normalization_.assign(storage_, 0.0);
    for (int angular_momentum = 0; angular_momentum <= multipole_order_;
         ++angular_momentum) {
        for (int magnetic = 0; magnetic <= angular_momentum; ++magnetic) {
            const double log_ratio =
                std::lgamma(static_cast<double>(angular_momentum - magnetic + 1)) -
                std::lgamma(static_cast<double>(angular_momentum + magnetic + 1));
            normalization_[angular_index(stride_, angular_momentum, magnetic)] =
                std::sqrt(
                    static_cast<double>(2 * angular_momentum + 1) *
                    std::exp(log_ratio) / (4.0 * kPi)
                );
        }
    }

    symmetry_representative_rows_.resize(wedge_count);
    symmetry_multiplicities_.resize(wedge_count);
    for (std::size_t wedge = 0; wedge < wedge_count; ++wedge) {
        const std::int64_t multiplicity = multiplicities.data()[wedge];
        if (multiplicity <= 0) {
            throw std::invalid_argument("multiplicities must be positive");
        }
        symmetry_representative_rows_[wedge] = static_cast<std::int64_t>(wedge);
        symmetry_multiplicities_[wedge] = multiplicity;
    }
    const std::size_t coefficient_count = wedge_count * angular_count;
    symmetry_moment_coefficients_.assign(
        moment_coefficients.data(),
        moment_coefficients.data() + coefficient_count
    );
    for (const auto value : symmetry_moment_coefficients_) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            throw std::invalid_argument(
                "moment_coefficients must contain finite values"
            );
        }
    }
    boundary_indptr_.assign(
        boundary_indptr.data(), boundary_indptr.data() + wedge_count + 1
    );
    boundary_operator_coefficient_.assign(
        boundary_operator_coefficient.data(),
        boundary_operator_coefficient.data() + term_count
    );
    boundary_radius_.assign(
        boundary_radius.data(), boundary_radius.data() + term_count
    );
    boundary_cosine_.assign(
        boundary_cosine.data(), boundary_cosine.data() + term_count
    );
    boundary_sine_.assign(
        boundary_sine.data(), boundary_sine.data() + term_count
    );
    boundary_phase_real_.assign(
        boundary_phase_real.data(), boundary_phase_real.data() + term_count
    );
    boundary_phase_imag_.assign(
        boundary_phase_imag.data(), boundary_phase_imag.data() + term_count
    );
    for (const double radius : boundary_radius_) {
        if (!(radius > 0.0)) {
            throw std::invalid_argument("boundary radii must be positive");
        }
    }
}

MultipoleBoundaryBuilder::MultipoleBoundaryBuilder(
    const IndexArray& integer_coordinates,
    const FloatArray& coordinates,
    const IndexArray& index_min,
    const IndexArray& lookup,
    const FloatArray& shift,
    int expansion_order,
    double spacing,
    int multipole_order
) {
    if (
        integer_coordinates.ndim() != 2 ||
        integer_coordinates.shape(1) != 3
    ) {
        throw std::invalid_argument(
            "integer_coordinates must have shape (n, 3)"
        );
    }
    if (
        coordinates.ndim() != 2 || coordinates.shape(1) != 3 ||
        coordinates.shape(0) != integer_coordinates.shape(0)
    ) {
        throw std::invalid_argument("coordinates must have shape (n, 3)");
    }
    if (index_min.ndim() != 1 || index_min.shape(0) != 3) {
        throw std::invalid_argument("index_min must have shape (3,)");
    }
    if (lookup.ndim() != 3) {
        throw std::invalid_argument("lookup must be a three-dimensional array");
    }
    if (shift.ndim() != 1 || shift.shape(0) != 3) {
        throw std::invalid_argument("shift must have shape (3,)");
    }
    if (!(spacing > 0.0) || !std::isfinite(spacing)) {
        throw std::invalid_argument("spacing must be finite and positive");
    }
    if (multipole_order < 0 || multipole_order > kMaximumMultipoleOrder) {
        throw std::invalid_argument("multipole_order must be between 0 and 60");
    }

    const std::vector<double> coefficients =
        second_derivative_coefficients(expansion_order);
    const int stencil_width = expansion_order / 2;
    const double inverse_spacing_squared = 1.0 / (spacing * spacing);
    point_count_ = static_cast<std::size_t>(coordinates.shape(0));
    multipole_order_ = multipole_order;
    stride_ = multipole_order + 1;
    storage_ = static_cast<std::size_t>(stride_) * static_cast<std::size_t>(stride_);
    volume_element_ = spacing * spacing * spacing;

    const auto integer = integer_coordinates.unchecked<2>();
    const auto physical = coordinates.unchecked<2>();
    const auto minimum = index_min.unchecked<1>();
    const auto lookup_values = lookup.unchecked<3>();
    const auto grid_shift = shift.unchecked<1>();

    std::array<std::int64_t, 3> index_minimum = {
        minimum(0), minimum(1), minimum(2)
    };
    std::array<std::int64_t, 3> lookup_shape = {
        lookup.shape(0), lookup.shape(1), lookup.shape(2)
    };
    std::array<double, 3> coordinate_shift = {
        grid_shift(0), grid_shift(1), grid_shift(2)
    };

    source_radius_.resize(point_count_);
    source_cosine_.resize(point_count_);
    source_sine_.resize(point_count_);
    source_phase_real_.resize(point_count_);
    source_phase_imag_.resize(point_count_);
    boundary_indptr_.resize(point_count_ + 1, 0);
    normalization_.assign(storage_, 0.0);

    for (int angular_momentum = 0; angular_momentum <= multipole_order_;
         ++angular_momentum) {
        for (int magnetic = 0; magnetic <= angular_momentum; ++magnetic) {
            const double log_ratio =
                std::lgamma(
                    static_cast<double>(angular_momentum - magnetic + 1)
                ) -
                std::lgamma(
                    static_cast<double>(angular_momentum + magnetic + 1)
                );
            normalization_[angular_index(stride_, angular_momentum, magnetic)] =
                std::sqrt(
                    static_cast<double>(2 * angular_momentum + 1) *
                    std::exp(log_ratio) / (4.0 * kPi)
                );
        }
    }

    // Copy all Python-owned values needed below before releasing the GIL.
    std::vector<std::int64_t> integer_copy(point_count_ * 3);
    std::vector<double> physical_copy(point_count_ * 3);
    for (std::size_t row = 0; row < point_count_; ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            integer_copy[row * 3 + axis] =
                integer(static_cast<py::ssize_t>(row), axis);
            physical_copy[row * 3 + axis] =
                physical(static_cast<py::ssize_t>(row), axis);
        }
    }
    const std::size_t lookup_size =
        static_cast<std::size_t>(lookup_shape[0]) *
        static_cast<std::size_t>(lookup_shape[1]) *
        static_cast<std::size_t>(lookup_shape[2]);
    std::vector<std::int64_t> lookup_copy(lookup_size);
    std::copy(lookup.data(), lookup.data() + lookup_size, lookup_copy.begin());

    py::gil_scoped_release release;

#pragma omp parallel for schedule(static) if(point_count_ >= 4096)
    for (std::int64_t row = 0;
         row < static_cast<std::int64_t>(point_count_); ++row) {
        const AngularGeometry geometry = angular_geometry(
            physical_copy[static_cast<std::size_t>(row) * 3],
            physical_copy[static_cast<std::size_t>(row) * 3 + 1],
            physical_copy[static_cast<std::size_t>(row) * 3 + 2]
        );
        source_radius_[static_cast<std::size_t>(row)] = geometry.radius;
        source_cosine_[static_cast<std::size_t>(row)] = geometry.cosine;
        source_sine_[static_cast<std::size_t>(row)] = geometry.sine;
        source_phase_real_[static_cast<std::size_t>(row)] = geometry.phase_real;
        source_phase_imag_[static_cast<std::size_t>(row)] = geometry.phase_imag;
    }

    // First count exterior stencil entries for every row.
#pragma omp parallel for schedule(static) if(point_count_ >= 4096)
    for (std::int64_t row = 0;
         row < static_cast<std::int64_t>(point_count_); ++row) {
        std::int64_t missing_count = 0;
        for (int axis = 0; axis < 3; ++axis) {
            for (int signed_shell = -stencil_width;
                 signed_shell <= stencil_width; ++signed_shell) {
                if (signed_shell == 0) {
                    continue;
                }
                std::array<std::int64_t, 3> local;
                bool inside_lookup = true;
                for (int dimension = 0; dimension < 3; ++dimension) {
                    const std::int64_t candidate =
                        integer_copy[static_cast<std::size_t>(row) * 3 + dimension] +
                        (dimension == axis ? signed_shell : 0);
                    local[dimension] = candidate - index_minimum[dimension];
                    inside_lookup = inside_lookup && local[dimension] >= 0 &&
                        local[dimension] < lookup_shape[dimension];
                }
                std::int64_t neighbor = -1;
                if (inside_lookup) {
                    const std::size_t lookup_index =
                        (
                            static_cast<std::size_t>(local[0]) *
                                static_cast<std::size_t>(lookup_shape[1]) +
                            static_cast<std::size_t>(local[1])
                        ) * static_cast<std::size_t>(lookup_shape[2]) +
                        static_cast<std::size_t>(local[2]);
                    neighbor = lookup_copy[lookup_index];
                }
                if (neighbor < 0) {
                    ++missing_count;
                }
            }
        }
        boundary_indptr_[static_cast<std::size_t>(row) + 1] = missing_count;
    }

    for (std::size_t row = 0; row < point_count_; ++row) {
        boundary_indptr_[row + 1] += boundary_indptr_[row];
    }
    const std::size_t term_count =
        static_cast<std::size_t>(boundary_indptr_.back());
    boundary_operator_coefficient_.resize(term_count);
    boundary_radius_.resize(term_count);
    boundary_cosine_.resize(term_count);
    boundary_sine_.resize(term_count);
    boundary_phase_real_.resize(term_count);
    boundary_phase_imag_.resize(term_count);

    // Fill each row's private section in the same axis/shell order used by
    // the Python reference boundary correction.
#pragma omp parallel for schedule(static) if(point_count_ >= 4096)
    for (std::int64_t row = 0;
         row < static_cast<std::int64_t>(point_count_); ++row) {
        std::size_t output = static_cast<std::size_t>(
            boundary_indptr_[static_cast<std::size_t>(row)]
        );
        for (int axis = 0; axis < 3; ++axis) {
            for (int signed_shell = -stencil_width;
                 signed_shell <= stencil_width; ++signed_shell) {
                if (signed_shell == 0) {
                    continue;
                }
                std::array<std::int64_t, 3> candidate;
                std::array<std::int64_t, 3> local;
                bool inside_lookup = true;
                for (int dimension = 0; dimension < 3; ++dimension) {
                    candidate[dimension] =
                        integer_copy[static_cast<std::size_t>(row) * 3 + dimension] +
                        (dimension == axis ? signed_shell : 0);
                    local[dimension] =
                        candidate[dimension] - index_minimum[dimension];
                    inside_lookup = inside_lookup && local[dimension] >= 0 &&
                        local[dimension] < lookup_shape[dimension];
                }
                std::int64_t neighbor = -1;
                if (inside_lookup) {
                    const std::size_t lookup_index =
                        (
                            static_cast<std::size_t>(local[0]) *
                                static_cast<std::size_t>(lookup_shape[1]) +
                            static_cast<std::size_t>(local[1])
                        ) * static_cast<std::size_t>(lookup_shape[2]) +
                        static_cast<std::size_t>(local[2]);
                    neighbor = lookup_copy[lookup_index];
                }
                if (neighbor >= 0) {
                    continue;
                }

                const int shell = std::abs(signed_shell);
                boundary_operator_coefficient_[output] =
                    -coefficients[stencil_width + shell] *
                    inverse_spacing_squared;
                const double x =
                    (static_cast<double>(candidate[0]) + coordinate_shift[0]) *
                    spacing;
                const double y =
                    (static_cast<double>(candidate[1]) + coordinate_shift[1]) *
                    spacing;
                const double z =
                    (static_cast<double>(candidate[2]) + coordinate_shift[2]) *
                    spacing;
                const AngularGeometry geometry = angular_geometry(x, y, z);
                if (!(geometry.radius > 0.0)) {
                    throw std::runtime_error(
                        "multipole exterior stencil contains the origin"
                    );
                }
                boundary_radius_[output] = geometry.radius;
                boundary_cosine_[output] = geometry.cosine;
                boundary_sine_[output] = geometry.sine;
                boundary_phase_real_[output] = geometry.phase_real;
                boundary_phase_imag_[output] = geometry.phase_imag;
                ++output;
            }
        }
    }
}

void MultipoleBoundaryBuilder::configure_symmetry(
    const IndexArray& representative_rows,
    const IndexArray& full_to_wedge,
    const IndexArray& multiplicities
) {
    if (
        representative_rows.ndim() != 1 ||
        multiplicities.ndim() != 1 ||
        representative_rows.shape(0) != multiplicities.shape(0) ||
        representative_rows.shape(0) < 1
    ) {
        throw std::invalid_argument(
            "representative_rows and multiplicities must be equal nonempty vectors"
        );
    }
    if (
        full_to_wedge.ndim() != 1 ||
        static_cast<std::size_t>(full_to_wedge.shape(0)) != point_count_
    ) {
        throw std::invalid_argument("full_to_wedge does not match the active grid");
    }
    const std::size_t wedge_count = static_cast<std::size_t>(
        representative_rows.shape(0)
    );
    const auto representatives = representative_rows.unchecked<1>();
    const auto mapping = full_to_wedge.unchecked<1>();
    const auto orbit_sizes = multiplicities.unchecked<1>();

    std::vector<std::int64_t> representative_copy(wedge_count);
    std::vector<std::int64_t> multiplicity_copy(wedge_count);
    std::vector<std::int64_t> mapping_copy(point_count_);
    std::vector<std::int64_t> counts(wedge_count, 0);
    for (std::size_t wedge = 0; wedge < wedge_count; ++wedge) {
        const std::int64_t representative = representatives(
            static_cast<py::ssize_t>(wedge)
        );
        const std::int64_t multiplicity = orbit_sizes(
            static_cast<py::ssize_t>(wedge)
        );
        if (
            representative < 0 ||
            representative >= static_cast<std::int64_t>(point_count_) ||
            multiplicity <= 0
        ) {
            throw std::invalid_argument("invalid symmetry representative or multiplicity");
        }
        representative_copy[wedge] = representative;
        multiplicity_copy[wedge] = multiplicity;
    }
    for (std::size_t row = 0; row < point_count_; ++row) {
        const std::int64_t wedge = mapping(static_cast<py::ssize_t>(row));
        if (wedge < 0 || wedge >= static_cast<std::int64_t>(wedge_count)) {
            throw std::invalid_argument("full_to_wedge contains an invalid orbit");
        }
        mapping_copy[row] = wedge;
        ++counts[static_cast<std::size_t>(wedge)];
    }
    for (std::size_t wedge = 0; wedge < wedge_count; ++wedge) {
        if (
            counts[wedge] != multiplicity_copy[wedge] ||
            mapping_copy[static_cast<std::size_t>(representative_copy[wedge])] !=
                static_cast<std::int64_t>(wedge)
        ) {
            throw std::invalid_argument("inconsistent symmetry orbit metadata");
        }
    }

    std::vector<std::int64_t> orbit_indptr(wedge_count + 1, 0);
    for (std::size_t wedge = 0; wedge < wedge_count; ++wedge) {
        orbit_indptr[wedge + 1] = orbit_indptr[wedge] + counts[wedge];
    }
    std::vector<std::int64_t> orbit_rows(point_count_);
    std::vector<std::int64_t> cursors = orbit_indptr;
    // Ascending full-row insertion makes precomputation deterministic across
    // OpenMP thread counts.
    for (std::size_t row = 0; row < point_count_; ++row) {
        const std::size_t wedge = static_cast<std::size_t>(mapping_copy[row]);
        orbit_rows[static_cast<std::size_t>(cursors[wedge]++)] =
            static_cast<std::int64_t>(row);
    }

    const std::size_t angular_count = static_cast<std::size_t>(
        (multipole_order_ + 1) * (multipole_order_ + 2) / 2
    );
    if (
        wedge_count >
        kMaximumSymmetryCoefficientBytes /
            (angular_count * sizeof(std::complex<double>))
    ) {
        throw std::runtime_error(
            "symmetry multipole coefficient cache would exceed 512 MiB"
        );
    }
    std::vector<std::complex<double>> coefficients(
        wedge_count * angular_count,
        std::complex<double>(0.0, 0.0)
    );
    {
        py::gil_scoped_release release;
#pragma omp parallel if(wedge_count >= 4096)
        {
        std::vector<std::complex<double>> harmonic(storage_);
        std::vector<double> radius_power(static_cast<std::size_t>(stride_));
#pragma omp for schedule(static)
        for (std::int64_t wedge_index = 0;
             wedge_index < static_cast<std::int64_t>(wedge_count);
             ++wedge_index) {
            const std::size_t wedge = static_cast<std::size_t>(wedge_index);
            std::complex<double>* output =
                coefficients.data() + wedge * angular_count;
            const std::size_t start = static_cast<std::size_t>(orbit_indptr[wedge]);
            const std::size_t stop = static_cast<std::size_t>(orbit_indptr[wedge + 1]);
            for (std::size_t position = start; position < stop; ++position) {
                const std::size_t row = static_cast<std::size_t>(
                    orbit_rows[position]
                );
                evaluate_positive_harmonics(
                    source_radius_[row],
                    source_cosine_[row],
                    source_sine_[row],
                    source_phase_real_[row],
                    source_phase_imag_[row],
                    multipole_order_,
                    normalization_,
                    harmonic,
                    radius_power
                );
                for (int angular_momentum = 0;
                     angular_momentum <= multipole_order_;
                     ++angular_momentum) {
                    const double radial_weight =
                        volume_element_ * radius_power[angular_momentum];
                    for (int magnetic = 0; magnetic <= angular_momentum;
                         ++magnetic) {
                        const std::size_t angular =
                            angular_index(stride_, angular_momentum, magnetic);
                        const std::size_t compact =
                            compact_angular_index(angular_momentum, magnetic);
                        output[compact] +=
                            radial_weight * std::conj(harmonic[angular]);
                    }
                }
            }
        }
        }
    }

    symmetry_representative_rows_ = std::move(representative_copy);
    symmetry_multiplicities_ = std::move(multiplicity_copy);
    symmetry_angular_count_ = angular_count;
    symmetry_moment_coefficients_ = std::move(coefficients);
}

py::dict MultipoleBoundaryBuilder::build_reduced(
    const FloatArray& wedge_density
) const {
    const std::size_t wedge_count = symmetry_representative_rows_.size();
    if (
        wedge_count == 0 || symmetry_angular_count_ == 0 ||
        symmetry_moment_coefficients_.empty()
    ) {
        throw std::runtime_error("symmetry metadata has not been configured");
    }
    if (
        wedge_density.ndim() != 1 ||
        static_cast<std::size_t>(wedge_density.shape(0)) != wedge_count
    ) {
        throw std::invalid_argument("density does not match the symmetry wedge");
    }
    const double* density_values = wedge_density.data();
    for (std::size_t wedge = 0; wedge < wedge_count; ++wedge) {
        if (!std::isfinite(density_values[wedge])) {
            throw std::invalid_argument("density must contain finite values");
        }
    }

    py::array_t<double> right_hand_side(static_cast<py::ssize_t>(wedge_count));
    py::array_t<std::complex<double>> moment_array({stride_, stride_});
    double* rhs = right_hand_side.mutable_data();
    std::vector<std::complex<double>> moments(
        storage_,
        std::complex<double>(0.0, 0.0)
    );
    {
        py::gil_scoped_release release;
        const std::size_t moment_block_count =
            (wedge_count + kMomentBlockSize - 1) / kMomentBlockSize;
        std::vector<std::complex<double>> partial_moments(
            moment_block_count * storage_,
            std::complex<double>(0.0, 0.0)
        );
#pragma omp parallel for schedule(static) if(wedge_count >= kMomentBlockSize) \
    num_threads(grid_vector_worker_count(wedge_count))
        for (std::int64_t block = 0;
             block < static_cast<std::int64_t>(moment_block_count);
             ++block) {
            std::complex<double>* local =
                partial_moments.data() +
                static_cast<std::size_t>(block) * storage_;
            const std::size_t start =
                static_cast<std::size_t>(block) * kMomentBlockSize;
            const std::size_t stop = std::min(
                wedge_count,
                start + kMomentBlockSize
            );
            for (std::size_t wedge = start; wedge < stop; ++wedge) {
                const std::complex<double>* coefficients =
                    symmetry_moment_coefficients_.data() +
                    wedge * symmetry_angular_count_;
                const double density_value = density_values[wedge];
                for (int angular_momentum = 0;
                     angular_momentum <= multipole_order_;
                     ++angular_momentum) {
                    for (int magnetic = 0; magnetic <= angular_momentum;
                         ++magnetic) {
                        const std::size_t angular =
                            angular_index(stride_, angular_momentum, magnetic);
                        const std::size_t compact =
                            compact_angular_index(angular_momentum, magnetic);
                        local[angular] += density_value * coefficients[compact];
                    }
                }
            }
        }
        for (std::size_t block = 0; block < moment_block_count; ++block) {
            const std::complex<double>* local =
                partial_moments.data() + block * storage_;
            for (int angular_momentum = 0;
                 angular_momentum <= multipole_order_;
                 ++angular_momentum) {
                for (int magnetic = 0; magnetic <= angular_momentum;
                     ++magnetic) {
                    const std::size_t angular =
                        angular_index(stride_, angular_momentum, magnetic);
                    moments[angular] += local[angular];
                }
            }
        }

#pragma omp parallel if(wedge_count >= 4096) \
    num_threads(grid_vector_worker_count(wedge_count))
        {
        std::vector<std::complex<double>> harmonic(storage_);
        std::vector<double> radius_power(static_cast<std::size_t>(stride_));
#pragma omp for schedule(static)
        for (std::int64_t wedge_index = 0;
             wedge_index < static_cast<std::int64_t>(wedge_count);
             ++wedge_index) {
            const std::size_t wedge = static_cast<std::size_t>(wedge_index);
            const std::size_t row = static_cast<std::size_t>(
                symmetry_representative_rows_[wedge]
            );
            double value = 8.0 * kPi * density_values[wedge];
            const std::size_t start = static_cast<std::size_t>(
                boundary_indptr_[row]
            );
            const std::size_t stop = static_cast<std::size_t>(
                boundary_indptr_[row + 1]
            );
            for (std::size_t term = start; term < stop; ++term) {
                const double potential = boundary_potential(
                    boundary_radius_[term],
                    boundary_cosine_[term],
                    boundary_sine_[term],
                    boundary_phase_real_[term],
                    boundary_phase_imag_[term],
                    multipole_order_,
                    normalization_,
                    moments,
                    harmonic,
                    radius_power
                );
                value -= boundary_operator_coefficient_[term] * potential;
            }
            // U.T applied to an invariant physical field multiplies its one
            // representative value by sqrt(orbit multiplicity).
            rhs[wedge] = value * std::sqrt(
                static_cast<double>(symmetry_multiplicities_[wedge])
            );
        }
        }
    }

    std::complex<double>* output_moments = moment_array.mutable_data();
    std::copy(moments.begin(), moments.end(), output_moments);
    py::dict result;
    result["right_hand_side"] = std::move(right_hand_side);
    result["positive_m_moments"] = std::move(moment_array);
    result["boundary_terms"] = py::int_(boundary_term_count());
    return result;
}

py::dict MultipoleBoundaryBuilder::export_symmetry_cache() const {
    const std::size_t wedge_count = symmetry_representative_rows_.size();
    if (
        wedge_count == 0 || symmetry_angular_count_ == 0 ||
        symmetry_moment_coefficients_.empty()
    ) {
        throw std::runtime_error("symmetry metadata has not been configured");
    }

    py::array_t<std::int64_t> multiplicities(
        static_cast<py::ssize_t>(wedge_count)
    );
    std::copy(
        symmetry_multiplicities_.begin(),
        symmetry_multiplicities_.end(),
        multiplicities.mutable_data()
    );
    py::array_t<std::complex<double>> moment_coefficients({
        static_cast<py::ssize_t>(wedge_count),
        static_cast<py::ssize_t>(symmetry_angular_count_)
    });
    std::copy(
        symmetry_moment_coefficients_.begin(),
        symmetry_moment_coefficients_.end(),
        moment_coefficients.mutable_data()
    );

    py::array_t<std::int64_t> wedge_indptr(
        static_cast<py::ssize_t>(wedge_count + 1)
    );
    auto* output_indptr = wedge_indptr.mutable_data();
    output_indptr[0] = 0;
    for (std::size_t wedge = 0; wedge < wedge_count; ++wedge) {
        const std::size_t row = static_cast<std::size_t>(
            symmetry_representative_rows_[wedge]
        );
        output_indptr[wedge + 1] = output_indptr[wedge] +
            boundary_indptr_[row + 1] - boundary_indptr_[row];
    }
    const std::size_t term_count = static_cast<std::size_t>(
        output_indptr[wedge_count]
    );
    py::array_t<double> coefficients(static_cast<py::ssize_t>(term_count));
    py::array_t<double> radii(static_cast<py::ssize_t>(term_count));
    py::array_t<double> cosines(static_cast<py::ssize_t>(term_count));
    py::array_t<double> sines(static_cast<py::ssize_t>(term_count));
    py::array_t<double> phase_real(static_cast<py::ssize_t>(term_count));
    py::array_t<double> phase_imag(static_cast<py::ssize_t>(term_count));
    std::size_t output = 0;
    for (std::size_t wedge = 0; wedge < wedge_count; ++wedge) {
        const std::size_t row = static_cast<std::size_t>(
            symmetry_representative_rows_[wedge]
        );
        const std::size_t start = static_cast<std::size_t>(boundary_indptr_[row]);
        const std::size_t stop = static_cast<std::size_t>(boundary_indptr_[row + 1]);
        for (std::size_t term = start; term < stop; ++term, ++output) {
            coefficients.mutable_data()[output] =
                boundary_operator_coefficient_[term];
            radii.mutable_data()[output] = boundary_radius_[term];
            cosines.mutable_data()[output] = boundary_cosine_[term];
            sines.mutable_data()[output] = boundary_sine_[term];
            phase_real.mutable_data()[output] = boundary_phase_real_[term];
            phase_imag.mutable_data()[output] = boundary_phase_imag_[term];
        }
    }

    py::dict result;
    result["multipole_order"] = multipole_order_;
    result["volume_element"] = volume_element_;
    result["multiplicities"] = std::move(multiplicities);
    result["moment_coefficients"] = std::move(moment_coefficients);
    result["boundary_indptr"] = std::move(wedge_indptr);
    result["boundary_operator_coefficient"] = std::move(coefficients);
    result["boundary_radius"] = std::move(radii);
    result["boundary_cosine"] = std::move(cosines);
    result["boundary_sine"] = std::move(sines);
    result["boundary_phase_real"] = std::move(phase_real);
    result["boundary_phase_imag"] = std::move(phase_imag);
    return result;
}

py::dict MultipoleBoundaryBuilder::export_full_geometry() const {
    if (source_radius_.size() != point_count_ || point_count_ == 0) {
        throw std::runtime_error("full-grid source geometry is unavailable");
    }
    py::dict result;
    auto copy_float = [&result](const char* name, const std::vector<double>& values) {
        py::array_t<double> output(static_cast<py::ssize_t>(values.size()));
        std::copy(values.begin(), values.end(), output.mutable_data());
        result[name] = std::move(output);
    };
    copy_float("source_radius", source_radius_);
    copy_float("source_cosine", source_cosine_);
    copy_float("source_sine", source_sine_);
    copy_float("source_phase_real", source_phase_real_);
    copy_float("source_phase_imag", source_phase_imag_);
    copy_float("boundary_radius", boundary_radius_);
    copy_float("boundary_cosine", boundary_cosine_);
    copy_float("boundary_sine", boundary_sine_);
    copy_float("boundary_phase_real", boundary_phase_real_);
    copy_float("boundary_phase_imag", boundary_phase_imag_);
    copy_float("boundary_operator_coefficient", boundary_operator_coefficient_);
    if (stride_ < kFormerAngularStride) {
        // Orders below 9 keep the layout that every Python tree reads: a
        // tree before 0.6.0 knows no other and would take its prefactors
        // from the wrong places without an error, the present one accepts
        // both.  At order 9 the two layouts are one.
        std::vector<double> former(
            static_cast<std::size_t>(kFormerAngularStride * kFormerAngularStride),
            0.0
        );
        for (int angular_momentum = 0; angular_momentum <= multipole_order_;
             ++angular_momentum) {
            for (int magnetic = 0; magnetic <= angular_momentum; ++magnetic) {
                former[angular_index(
                    kFormerAngularStride, angular_momentum, magnetic
                )] = normalization_[
                    angular_index(stride_, angular_momentum, magnetic)
                ];
            }
        }
        copy_float("normalization", former);
    } else {
        copy_float("normalization", normalization_);
    }
    py::array_t<std::int64_t> indptr(static_cast<py::ssize_t>(boundary_indptr_.size()));
    std::copy(boundary_indptr_.begin(), boundary_indptr_.end(), indptr.mutable_data());
    result["boundary_indptr"] = std::move(indptr);
    return result;
}

py::dict MultipoleBoundaryBuilder::build(const FloatArray& density) const {
    if (
        density.ndim() != 1 ||
        static_cast<std::size_t>(density.shape(0)) != point_count_
    ) {
        throw std::invalid_argument("density does not match the active grid");
    }
    const double* density_values = density.data();
    for (std::size_t row = 0; row < point_count_; ++row) {
        if (!std::isfinite(density_values[row])) {
            throw std::invalid_argument("density must contain finite values");
        }
    }

    py::array_t<double> right_hand_side(
        static_cast<py::ssize_t>(point_count_)
    );
    py::array_t<std::complex<double>> moment_array({stride_, stride_});
    double* rhs = right_hand_side.mutable_data();
    std::vector<std::complex<double>> moments(
        storage_,
        std::complex<double>(0.0, 0.0)
    );

    {
        py::gil_scoped_release release;

        const std::size_t moment_block_count =
            (point_count_ + kMomentBlockSize - 1) / kMomentBlockSize;
        std::vector<std::complex<double>> partial_moments(
            moment_block_count * storage_,
            std::complex<double>(0.0, 0.0)
        );

#pragma omp parallel if(point_count_ >= kMomentBlockSize) \
    num_threads(grid_vector_worker_count(point_count_))
        {
        std::vector<std::complex<double>> harmonic(storage_);
        std::vector<double> radius_power(static_cast<std::size_t>(stride_));
#pragma omp for schedule(static)
        for (std::int64_t block = 0;
             block < static_cast<std::int64_t>(moment_block_count); ++block) {
            std::complex<double>* local =
                partial_moments.data() +
                static_cast<std::size_t>(block) * storage_;
            const std::size_t start =
                static_cast<std::size_t>(block) * kMomentBlockSize;
            const std::size_t stop =
                std::min(point_count_, start + kMomentBlockSize);
            for (std::size_t index = start; index < stop; ++index) {
                evaluate_positive_harmonics(
                    source_radius_[index],
                    source_cosine_[index],
                    source_sine_[index],
                    source_phase_real_[index],
                    source_phase_imag_[index],
                    multipole_order_,
                    normalization_,
                    harmonic,
                    radius_power
                );
                const double weight = density_values[index] * volume_element_;
                for (int angular_momentum = 0;
                     angular_momentum <= multipole_order_; ++angular_momentum) {
                    const double radial_weight =
                        weight * radius_power[angular_momentum];
                    for (int magnetic = 0; magnetic <= angular_momentum;
                         ++magnetic) {
                        const std::size_t angular =
                            angular_index(stride_, angular_momentum, magnetic);
                        local[angular] +=
                            radial_weight * std::conj(harmonic[angular]);
                    }
                }
            }
        }
        }

        // Fixed source blocks and a block-order merge make the moments
        // independent of the number of OpenMP workers.
        for (std::size_t block = 0; block < moment_block_count; ++block) {
            const std::complex<double>* local =
                partial_moments.data() + block * storage_;
            for (int angular_momentum = 0;
                 angular_momentum <= multipole_order_; ++angular_momentum) {
                for (int magnetic = 0; magnetic <= angular_momentum;
                     ++magnetic) {
                    const std::size_t angular =
                        angular_index(stride_, angular_momentum, magnetic);
                    moments[angular] += local[angular];
                }
            }
        }

#pragma omp parallel if(point_count_ >= 4096) \
    num_threads(grid_vector_worker_count(point_count_))
        {
        std::vector<std::complex<double>> harmonic(storage_);
        std::vector<double> radius_power(static_cast<std::size_t>(stride_));
#pragma omp for schedule(static)
        for (std::int64_t row = 0;
             row < static_cast<std::int64_t>(point_count_); ++row) {
            const std::size_t index = static_cast<std::size_t>(row);
            double value = 8.0 * kPi * density_values[index];
            const std::size_t start =
                static_cast<std::size_t>(boundary_indptr_[index]);
            const std::size_t stop =
                static_cast<std::size_t>(boundary_indptr_[index + 1]);
            for (std::size_t term = start; term < stop; ++term) {
                const double potential = boundary_potential(
                    boundary_radius_[term],
                    boundary_cosine_[term],
                    boundary_sine_[term],
                    boundary_phase_real_[term],
                    boundary_phase_imag_[term],
                    multipole_order_,
                    normalization_,
                    moments,
                    harmonic,
                    radius_power
                );
                value -= boundary_operator_coefficient_[term] * potential;
            }
            rhs[index] = value;
        }
        }
    }

    std::complex<double>* output_moments = moment_array.mutable_data();
    std::copy(moments.begin(), moments.end(), output_moments);
    py::dict result;
    result["right_hand_side"] = std::move(right_hand_side);
    result["positive_m_moments"] = std::move(moment_array);
    result["boundary_terms"] = py::int_(boundary_term_count());
    return result;
}

std::size_t MultipoleBoundaryBuilder::size() const noexcept {
    return point_count_;
}

std::size_t MultipoleBoundaryBuilder::boundary_term_count() const noexcept {
    return boundary_operator_coefficient_.size();
}

int MultipoleBoundaryBuilder::multipole_order() const noexcept {
    return multipole_order_;
}

std::size_t MultipoleBoundaryBuilder::symmetry_wedge_size() const noexcept {
    return symmetry_representative_rows_.size();
}

}  // namespace parsec_accelerated_native
