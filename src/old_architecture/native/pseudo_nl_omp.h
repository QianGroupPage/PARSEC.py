#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace rsdft_native {

namespace py = pybind11;

py::object pseudo_nl_omp(
    const py::dict& domain,
    const py::list& species
);

}  // namespace rsdft_native
