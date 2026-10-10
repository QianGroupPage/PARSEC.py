#include "finite_difference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace parsec_accelerated_native {
namespace {

template <typename T>
py::array_t<T> vector_to_array(const std::vector<T>& values) {
    py::array_t<T> result(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(
            result.mutable_data(),
            values.data(),
            values.size() * sizeof(T)
        );
    }
    return result;
}

std::uint64_t factorial(const int value) {
    std::uint64_t result = 1;
    for (int factor = 2; factor <= value; ++factor) {
        result *= static_cast<std::uint64_t>(factor);
    }
    return result;
}

}  // namespace

std::vector<double> centered_second_derivative_coefficients(
    const int expansion_order
) {
    if (
        expansion_order < 2 || expansion_order > 20 ||
        expansion_order % 2 != 0
    ) {
        throw std::invalid_argument(
            "expansion_order must be an even integer from 2 to 20"
        );
    }

    const int width = expansion_order / 2;
    std::vector<double> coefficients(
        static_cast<std::size_t>(2 * width + 1),
        0.0
    );
    const double width_factorial = static_cast<double>(factorial(width));
    double positive_sum = 0.0;
    for (int shell = 1; shell <= width; ++shell) {
        const double sign = (shell % 2 == 1) ? 1.0 : -1.0;
        const double denominator =
            static_cast<double>(shell * shell) *
            static_cast<double>(factorial(width - shell)) *
            static_cast<double>(factorial(width + shell));
        const double value =
            2.0 * sign * width_factorial * width_factorial / denominator;
        coefficients[static_cast<std::size_t>(width - shell)] = value;
        coefficients[static_cast<std::size_t>(width + shell)] = value;
        positive_sum += value;
    }
    coefficients[static_cast<std::size_t>(width)] = -2.0 * positive_sum;
    return coefficients;
}

py::dict build_negative_laplacian_buffers(
    const IndexArray& integer_coordinates,
    const IndexArray& index_min,
    const IndexArray& lookup,
    const int expansion_order,
    const double spacing
) {
    if (
        integer_coordinates.ndim() != 2 ||
        integer_coordinates.shape(1) != 3
    ) {
        throw std::invalid_argument(
            "integer_coordinates must have shape (number_of_points, 3)"
        );
    }
    if (index_min.ndim() != 1 || index_min.shape(0) != 3) {
        throw std::invalid_argument("index_min must contain three integers");
    }
    if (lookup.ndim() != 3) {
        throw std::invalid_argument("lookup must be a three-dimensional array");
    }
    if (!std::isfinite(spacing) || spacing <= 0.0) {
        throw std::invalid_argument("spacing must be positive and finite");
    }

    const auto coefficients =
        centered_second_derivative_coefficients(expansion_order);
    const int width = expansion_order / 2;
    const double inverse_spacing_squared = 1.0 / (spacing * spacing);
    const std::int64_t point_count =
        static_cast<std::int64_t>(integer_coordinates.shape(0));
    if (point_count < 1) {
        throw std::invalid_argument("the compressed grid cannot be empty");
    }

    const auto* coordinates = integer_coordinates.data();
    const auto* minima = index_min.data();
    const auto* lookup_data = lookup.data();
    const std::int64_t lookup_shape[3] = {
        static_cast<std::int64_t>(lookup.shape(0)),
        static_cast<std::int64_t>(lookup.shape(1)),
        static_cast<std::int64_t>(lookup.shape(2)),
    };

    std::vector<std::int64_t> indptr(
        static_cast<std::size_t>(point_count + 1),
        0
    );
    std::vector<std::int64_t> indices;
    std::vector<double> data;
    const std::size_t maximum_row_entries =
        static_cast<std::size_t>(1 + 6 * width);
    indices.reserve(static_cast<std::size_t>(point_count) * maximum_row_entries);
    data.reserve(static_cast<std::size_t>(point_count) * maximum_row_entries);

    auto row_for_point = [&](const std::int64_t x, const std::int64_t y,
                             const std::int64_t z) -> std::int64_t {
        const std::int64_t local_x = x - minima[0];
        const std::int64_t local_y = y - minima[1];
        const std::int64_t local_z = z - minima[2];
        if (
            local_x < 0 || local_x >= lookup_shape[0] ||
            local_y < 0 || local_y >= lookup_shape[1] ||
            local_z < 0 || local_z >= lookup_shape[2]
        ) {
            return -1;
        }
        const std::int64_t offset =
            (local_x * lookup_shape[1] + local_y) * lookup_shape[2] + local_z;
        return lookup_data[offset];
    };

    {
        py::gil_scoped_release release;
        for (std::int64_t row = 0; row < point_count; ++row) {
            const auto base = static_cast<std::size_t>(3 * row);
            const std::int64_t point[3] = {
                coordinates[base],
                coordinates[base + 1],
                coordinates[base + 2],
            };
            if (row_for_point(point[0], point[1], point[2]) != row) {
                throw std::invalid_argument(
                    "lookup does not map integer_coordinates back to their rows"
                );
            }

            std::vector<std::pair<std::int64_t, double>> row_entries;
            row_entries.reserve(maximum_row_entries);
            row_entries.emplace_back(
                row,
                -3.0 * coefficients[static_cast<std::size_t>(width)] *
                    inverse_spacing_squared
            );

            // Preserve PARSEC/Python shell enumeration.  Sorting by the final
            // column below produces canonical CSR and fixes apply order.
            for (int axis = 0; axis < 3; ++axis) {
                for (int signed_shell = -width; signed_shell <= width;
                     ++signed_shell) {
                    if (signed_shell == 0) {
                        continue;
                    }
                    std::int64_t displaced[3] = {point[0], point[1], point[2]};
                    displaced[axis] += signed_shell;
                    const std::int64_t neighbor = row_for_point(
                        displaced[0], displaced[1], displaced[2]
                    );
                    if (neighbor < 0) {
                        continue;
                    }
                    if (neighbor >= point_count) {
                        throw std::invalid_argument(
                            "lookup contains an active row outside the grid"
                        );
                    }
                    const int shell = std::abs(signed_shell);
                    row_entries.emplace_back(
                        neighbor,
                        -coefficients[static_cast<std::size_t>(width + shell)] *
                            inverse_spacing_squared
                    );
                }
            }

            std::stable_sort(
                row_entries.begin(),
                row_entries.end(),
                [](const auto& left, const auto& right) {
                    return left.first < right.first;
                }
            );
            for (const auto& entry : row_entries) {
                indices.push_back(entry.first);
                data.push_back(entry.second);
            }
            indptr[static_cast<std::size_t>(row + 1)] =
                static_cast<std::int64_t>(indices.size());
        }
    }

    py::dict result;
    result["indptr"] = vector_to_array(indptr);
    result["indices"] = vector_to_array(indices);
    result["data"] = vector_to_array(data);
    result["shape"] = py::make_tuple(point_count, point_count);
    result["nnz"] = py::int_(static_cast<std::int64_t>(data.size()));
    result["expansion_order"] = expansion_order;
    return result;
}

}  // namespace parsec_accelerated_native
