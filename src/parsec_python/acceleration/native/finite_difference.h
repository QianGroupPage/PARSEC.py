#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace parsec_accelerated_native {

namespace py = pybind11;

using IndexArray = py::array_t<
    std::int64_t,
    py::array::c_style | py::array::forcecast
>;
using FloatArray = py::array_t<
    double,
    py::array::c_style | py::array::forcecast
>;

// Centered weights [c_-M, ..., c_0, ..., c_M] of the positive second
// derivative.  Every builder of the stencil takes them from here, so that
// each one stores the same float64 values.
std::vector<double> centered_second_derivative_coefficients(
    int expansion_order
);

py::dict build_negative_laplacian_buffers(
    const IndexArray& integer_coordinates,
    const IndexArray& index_min,
    const IndexArray& lookup,
    int expansion_order,
    double spacing
);

}  // namespace parsec_accelerated_native
