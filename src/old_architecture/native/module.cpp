#include "pseudo_diag_omp.h"
#include "pseudo_nl_omp.h"

#include <pybind11/pybind11.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

PYBIND11_MODULE(rsdft_native, m) {
    m.doc() =
        "Scaffold module for future C++/OpenMP acceleration of PARSEC.py ionic setup kernels.";

    m.def("build_info", []() {
        py::dict info;
        info["scaffold"] = false;
        info["implemented_kernels"] = py::make_tuple("pseudo_diag_omp", "pseudo_nl_omp");
#ifdef _OPENMP
        info["openmp_enabled"] = true;
        info["openmp_max_threads"] = omp_get_max_threads();
#else
        info["openmp_enabled"] = false;
        info["openmp_max_threads"] = 1;
#endif
        return info;
    });

    m.def(
        "pseudo_diag_omp",
        &rsdft_native::pseudo_diag_omp,
        py::arg("domain"),
        py::arg("species"),
        py::arg("z_sum"),
        py::arg("return_info") = false,
        py::arg("build_hpot") = true,
        R"pbdoc(
Native entry point for the diagonal ionic setup hot loop.

The Python wrapper is responsible for loading splineData.mat, preprocessing
the radial tables, and constructing plain NumPy buffers for each species.
This function copies those inputs into C++ buffers, releases the GIL around
the row-wise grid loops, and parallelizes those loops with OpenMP.
)pbdoc"
    );

    m.def(
        "pseudo_nl_omp",
        &rsdft_native::pseudo_nl_omp,
        py::arg("domain"),
        py::arg("species"),
        R"pbdoc(
Native entry point for the nonlocal ionic setup hot loop.

The Python wrapper is responsible for loading splineData.mat, preprocessing
the radial tables, and constructing plain NumPy buffers for each species.
This function copies those inputs into C++ buffers, releases the GIL around
the atom-local projector assembly, and returns COO triplets for SciPy sparse
matrix construction on the Python side.
)pbdoc"
    );
}
