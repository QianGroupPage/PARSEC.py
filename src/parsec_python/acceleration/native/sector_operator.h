#pragma once
#include "finite_difference.h"

namespace parsec_accelerated_native {
py::dict reduce_sector_csr(
    const py::array& indptr, const py::array& indices, const FloatArray& data,
    const IndexArray& representatives, const IndexArray& full_to_orbit,
    const IndexArray& orbit_to_sector, const IndexArray& multiplicities,
    const py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>& phases);
}
