#pragma once

#include "finite_difference.h"

namespace parsec_accelerated_native {

py::dict build_sector_stencil(
    const IndexArray& integer_coordinates,
    const IndexArray& index_min,
    const IndexArray& lookup,
    int expansion_order,
    double spacing,
    const IndexArray& representatives,
    const IndexArray& full_to_orbit,
    const IndexArray& orbit_to_sector,
    const IndexArray& multiplicities,
    const py::array_t<
        std::int8_t,
        py::array::c_style | py::array::forcecast
    >& phases,
    int threads = 0
);

}  // namespace parsec_accelerated_native
