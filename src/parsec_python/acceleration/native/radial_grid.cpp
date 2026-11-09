#include "radial_grid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace parsec_accelerated_native {
namespace {

void validate_position(const FloatArray& position) {
    if (position.ndim() != 1 || position.shape(0) != 3) {
        throw std::invalid_argument("atom_position must have shape (3,)");
    }
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(position.data()[axis])) {
            throw std::invalid_argument("atom_position must be finite");
        }
    }
}

void validate_radial_table(
    const FloatArray& radii,
    const FloatArray& values
) {
    if (
        radii.ndim() != 1 || values.ndim() != 1 || radii.size() < 2 ||
        values.size() != radii.size()
    ) {
        throw std::invalid_argument(
            "radii and values must be equal one-dimensional radial tables"
        );
    }
    const double* radial = radii.data();
    const double* field = values.data();
    for (py::ssize_t index = 0; index < radii.size(); ++index) {
        if (
            !std::isfinite(radial[index]) || radial[index] <= 0.0 ||
            !std::isfinite(field[index]) ||
            (index > 0 && radial[index] <= radial[index - 1])
        ) {
            throw std::invalid_argument(
                "radial knots must be finite, positive, and strictly increasing"
            );
        }
    }
}

bool spline_enabled(
    const FloatArray& knots,
    const FloatArray& values,
    const FloatArray& second
) {
    const bool any = knots.size() || values.size() || second.size();
    if (!any) {
        return false;
    }
    if (
        knots.ndim() != 1 || values.ndim() != 1 || second.ndim() != 1 ||
        knots.size() < 2 || values.size() != knots.size() ||
        second.size() != knots.size()
    ) {
        throw std::invalid_argument(
            "spline knots, values, and second derivatives must have equal size"
        );
    }
    const double* x = knots.data();
    for (py::ssize_t index = 0; index < knots.size(); ++index) {
        if (
            !std::isfinite(x[index]) || !std::isfinite(values.data()[index]) ||
            !std::isfinite(second.data()[index]) ||
            (index > 0 && x[index] <= x[index - 1])
        ) {
            throw std::invalid_argument("invalid radial spline table");
        }
    }
    return true;
}

std::size_t lower_interval(
    const double* knots,
    std::size_t count,
    double query
) {
    const double* upper = std::upper_bound(knots, knots + count, query);
    if (upper <= knots) {
        return 0;
    }
    if (upper >= knots + count) {
        return count - 2;
    }
    return static_cast<std::size_t>(upper - knots - 1);
}

double linear_interpolate(
    const double* knots,
    const double* values,
    std::size_t count,
    double query
) {
    if (query <= knots[0]) {
        return values[0];
    }
    if (query >= knots[count - 1]) {
        return values[count - 1];
    }
    const std::size_t lower = lower_interval(knots, count, query);
    const double fraction =
        (query - knots[lower]) / (knots[lower + 1] - knots[lower]);
    return values[lower] +
        fraction * (values[lower + 1] - values[lower]);
}

double spline_interpolate(
    const double* knots,
    const double* values,
    const double* second,
    std::size_t count,
    double query
) {
    const std::size_t lower = lower_interval(knots, count, query);
    const std::size_t upper = lower + 1;
    const double step = knots[upper] - knots[lower];
    const double left = (knots[upper] - query) / step;
    const double right = (query - knots[lower]) / step;
    return left * values[lower] + right * values[upper] +
        (
            (left * left * left - left) * second[lower] +
            (right * right * right - right) * second[upper]
        ) * step * step / 6.0;
}

std::array<double, 7> real_harmonics(
    int angular_momentum,
    double x,
    double y,
    double z,
    double radius
) {
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;
    if (radius > 0.0) {
        ux = x / radius;
        uy = y / radius;
        uz = z / radius;
    }
    std::array<double, 7> result{};
    if (angular_momentum == 0) {
        result[0] = 0.28209479177387814;
    } else if (angular_momentum == 1) {
        constexpr double c = 0.4886025119029199;
        result[0] = c * ux;
        result[1] = c * uy;
        result[2] = c * uz;
    } else if (angular_momentum == 2) {
        constexpr double c = 1.0925484305920792;
        result[0] = c * ux * uy;
        result[1] = c * uy * uz;
        result[2] = c * ux * uz;
        result[3] = 0.31539156525252005 * (3.0 * uz * uz - 1.0);
        result[4] = 0.5 * c * (ux * ux - uy * uy);
    } else if (angular_momentum == 3) {
        result[0] = 0.5900435899266435 * uy * (3.0 * ux * ux - uy * uy);
        result[1] = 2.890611442640554 * ux * uy * uz;
        result[2] = 0.4570457994644658 * uy * (5.0 * uz * uz - 1.0);
        result[3] = 0.3731763325901154 * uz * (5.0 * uz * uz - 3.0);
        result[4] = 0.4570457994644658 * ux * (5.0 * uz * uz - 1.0);
        result[5] = 1.445305721320277 * uz * (ux * ux - uy * uy);
        result[6] = 0.5900435899266435 * ux * (ux * ux - 3.0 * uy * uy);
    } else {
        throw std::invalid_argument("angular_momentum must be between 0 and 3");
    }
    return result;
}

}  // namespace

RadialGridEvaluator::RadialGridEvaluator(const FloatArray& coordinates) :
    coordinates_(
        coordinates.data(),
        coordinates.data() + static_cast<std::size_t>(coordinates.size())
    ) {
    if (coordinates.ndim() != 2 || coordinates.shape(1) != 3) {
        throw std::invalid_argument("coordinates must have shape (N, 3)");
    }
    if (!std::all_of(
            coordinates_.begin(), coordinates_.end(),
            [](double value) { return std::isfinite(value); }
        )) {
        throw std::invalid_argument("coordinates must be finite");
    }
}

FloatArray RadialGridEvaluator::local_potential(
    const FloatArray& atom_position,
    const FloatArray& radii,
    const FloatArray& values,
    double ionic_charge,
    const FloatArray& spline_knots,
    const FloatArray& spline_values,
    const FloatArray& spline_second_derivatives
) const {
    validate_position(atom_position);
    validate_radial_table(radii, values);
    if (!std::isfinite(ionic_charge)) {
        throw std::invalid_argument("ionic_charge must be finite");
    }
    const bool use_spline = spline_enabled(
        spline_knots, spline_values, spline_second_derivatives
    );
    const std::size_t count = size();
    py::array_t<double> result(static_cast<py::ssize_t>(count));
    double* output = result.mutable_data();
    const double* position = atom_position.data();
    const double* radial = radii.data();
    const double* field = values.data();
    const std::size_t radial_count = static_cast<std::size_t>(radii.size());
    const double cutoff = radial[radial_count - 2];

    std::vector<double> radius_times_value;
    if (!use_spline) {
        radius_times_value.resize(radial_count);
        for (std::size_t index = 0; index < radial_count; ++index) {
            radius_times_value[index] = radial[index] * field[index];
        }
    }
    {
        py::gil_scoped_release release;
#pragma omp parallel for schedule(static) if(count >= 4096)
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(count); ++index) {
            const std::size_t row = static_cast<std::size_t>(index);
            const double x = coordinates_[3 * row] - position[0];
            const double y = coordinates_[3 * row + 1] - position[1];
            const double z = coordinates_[3 * row + 2] - position[2];
            const double distance = std::sqrt(x * x + y * y + z * z);
            if (distance >= cutoff) {
                output[row] = -2.0 * ionic_charge / distance;
            } else if (use_spline) {
                output[row] = spline_interpolate(
                    spline_knots.data(), spline_values.data(),
                    spline_second_derivatives.data(),
                    static_cast<std::size_t>(spline_knots.size()), distance
                );
            } else if (distance <= radial[0]) {
                output[row] = field[0];
            } else {
                output[row] = linear_interpolate(
                    radial, radius_times_value.data(), radial_count, distance
                ) / distance;
            }
        }
    }
    return result;
}

FloatArray RadialGridEvaluator::density(
    const FloatArray& atom_position,
    const FloatArray& radii,
    const FloatArray& values,
    const FloatArray& spline_knots,
    const FloatArray& spline_values,
    const FloatArray& spline_second_derivatives
) const {
    validate_position(atom_position);
    validate_radial_table(radii, values);
    const bool use_spline = spline_enabled(
        spline_knots, spline_values, spline_second_derivatives
    );
    const std::size_t count = size();
    py::array_t<double> result(static_cast<py::ssize_t>(count));
    double* output = result.mutable_data();
    const double* position = atom_position.data();
    const double* radial = radii.data();
    const std::size_t radial_count = static_cast<std::size_t>(radii.size());
    const double cutoff = radial[radial_count - 2];
    {
        py::gil_scoped_release release;
#pragma omp parallel for schedule(static) if(count >= 4096)
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(count); ++index) {
            const std::size_t row = static_cast<std::size_t>(index);
            const double x = coordinates_[3 * row] - position[0];
            const double y = coordinates_[3 * row + 1] - position[1];
            const double z = coordinates_[3 * row + 2] - position[2];
            const double distance = std::sqrt(x * x + y * y + z * z);
            if (distance >= cutoff) {
                output[row] = 0.0;
            } else if (use_spline) {
                output[row] = spline_interpolate(
                    spline_knots.data(), spline_values.data(),
                    spline_second_derivatives.data(),
                    static_cast<std::size_t>(spline_knots.size()), distance
                );
            } else {
                output[row] = linear_interpolate(
                    radial, values.data(), radial_count, distance
                );
            }
        }
    }
    return result;
}

py::dict RadialGridEvaluator::projector_channel(
    const FloatArray& atom_position,
    const FloatArray& radii,
    const FloatArray& radial_projector,
    double support_radius,
    int angular_momentum,
    double square_root_volume,
    const FloatArray& spline_knots,
    const FloatArray& spline_values,
    const FloatArray& spline_second_derivatives
) const {
    validate_position(atom_position);
    validate_radial_table(radii, radial_projector);
    if (!(support_radius >= 0.0) || !std::isfinite(support_radius)) {
        throw std::invalid_argument("support_radius must be finite and nonnegative");
    }
    if (!(square_root_volume > 0.0) || !std::isfinite(square_root_volume)) {
        throw std::invalid_argument("square_root_volume must be finite and positive");
    }
    if (angular_momentum < 0 || angular_momentum > 3) {
        throw std::invalid_argument("angular_momentum must be between 0 and 3");
    }
    const bool use_spline = spline_enabled(
        spline_knots, spline_values, spline_second_derivatives
    );
    const std::size_t count = size();
    const double* position = atom_position.data();

    std::vector<std::int64_t> support_rows;
    support_rows.reserve(count / 16 + 1);
    for (std::size_t row = 0; row < count; ++row) {
        const double x = coordinates_[3 * row] - position[0];
        const double y = coordinates_[3 * row + 1] - position[1];
        const double z = coordinates_[3 * row + 2] - position[2];
        if (x * x + y * y + z * z <= support_radius * support_radius) {
            support_rows.push_back(static_cast<std::int64_t>(row));
        }
    }

    const std::size_t support_count = support_rows.size();
    const int harmonic_count = 2 * angular_momentum + 1;
    py::array_t<std::int64_t> rows_array(static_cast<py::ssize_t>(support_count));
    py::array_t<double> values_array(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(support_count),
            static_cast<py::ssize_t>(harmonic_count),
        }
    );
    std::copy(support_rows.begin(), support_rows.end(), rows_array.mutable_data());
    double* output = values_array.mutable_data();
    const double* radial = radii.data();
    const std::size_t radial_count = static_cast<std::size_t>(radii.size());
    {
        py::gil_scoped_release release;
#pragma omp parallel for schedule(static) if(support_count >= 4096)
        for (std::int64_t local_index = 0;
             local_index < static_cast<std::int64_t>(support_count); ++local_index) {
            const std::size_t local = static_cast<std::size_t>(local_index);
            const std::size_t row = static_cast<std::size_t>(support_rows[local]);
            const double x = coordinates_[3 * row] - position[0];
            const double y = coordinates_[3 * row + 1] - position[1];
            const double z = coordinates_[3 * row + 2] - position[2];
            const double distance = std::sqrt(x * x + y * y + z * z);
            double radial_value = 0.0;
            if (use_spline) {
                radial_value = spline_interpolate(
                    spline_knots.data(), spline_values.data(),
                    spline_second_derivatives.data(),
                    static_cast<std::size_t>(spline_knots.size()),
                    std::max(distance, radial[0])
                );
            } else {
                radial_value = linear_interpolate(
                    radial, radial_projector.data(), radial_count, distance
                );
            }
            // A point on the atom up to round-off takes the zero-limit direction of the center: the cosines are
            // x / max(|d|, min(r_0, 1e-6 bohr)), the rule of the Python projectors and of the forces
            // (V_ion.ionic_potential.PROJECTOR_CENTER_ROUND_OFF, Forces.periodic.ROUND_OFF_DISTANCE).
            const double center_radius = std::min(radial[0], kProjectorCenterRoundOff);
            const auto harmonics = real_harmonics(
                angular_momentum, x, y, z,
                distance > 0.0 ? std::max(distance, center_radius) : 0.0
            );
            for (int harmonic = 0; harmonic < harmonic_count; ++harmonic) {
                output[local * static_cast<std::size_t>(harmonic_count) + harmonic] =
                    square_root_volume * radial_value * harmonics[harmonic];
            }
        }
    }

    py::dict result;
    result["rows"] = std::move(rows_array);
    result["values"] = std::move(values_array);
    return result;
}

std::size_t RadialGridEvaluator::size() const noexcept {
    return coordinates_.size() / 3;
}

}  // namespace parsec_accelerated_native
