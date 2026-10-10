#include "conjugate_gradient.h"
#include "ca_lda.h"
#include "finite_difference.h"
#include "fused_hamiltonian.h"
#include "hartree_boundary.h"
#include "radial_grid.h"
#include "sector_operator.h"
#include "sector_stencil.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using parsec_accelerated_native::FloatArray;
using parsec_accelerated_native::ConjugateGradientSolver;
using parsec_accelerated_native::CALDAEvaluator;
using parsec_accelerated_native::FusedHamiltonian;
using parsec_accelerated_native::IndexArray;
using parsec_accelerated_native::MultipoleBoundaryBuilder;
using parsec_accelerated_native::RadialGridEvaluator;
using parsec_accelerated_native::build_negative_laplacian_buffers;

namespace {

constexpr int kRequestedReservedThreads = 4;

struct OpenMPConfiguration {
    bool enabled = false;
    int detected_processors = 1;
    int reserved_threads = 0;
    int default_threads = 1;
    int max_threads = 1;
    bool dynamic = false;
    std::string source = "serial";
};

OpenMPConfiguration configure_openmp() {
    OpenMPConfiguration configuration;
#ifdef _OPENMP
    configuration.enabled = true;
    configuration.detected_processors = std::max(1, omp_get_num_procs());
    configuration.default_threads = std::max(
        1,
        configuration.detected_processors - kRequestedReservedThreads
    );
    configuration.reserved_threads =
        configuration.detected_processors - configuration.default_threads;

    // Respect the standard user override.  Otherwise reserve four logical
    // processors for the OS, Python orchestration, and CUDA driver activity.
    // Disabling dynamic adjustment makes the reported default predictable.
    const char* explicit_threads = std::getenv("OMP_NUM_THREADS");
    if (explicit_threads != nullptr && explicit_threads[0] != '\0') {
        configuration.source = "OMP_NUM_THREADS";
    } else {
        omp_set_dynamic(0);
        omp_set_num_threads(configuration.default_threads);
        configuration.source = "detected_processors_minus_4";
    }
    configuration.max_threads = omp_get_max_threads();
    configuration.dynamic = omp_get_dynamic() != 0;
#endif
    return configuration;
}

}  // namespace

PYBIND11_MODULE(parsec_accelerated_native, module) {
    const OpenMPConfiguration openmp = configure_openmp();
    module.doc() =
        "Float64 C++/OpenMP kernels for the accelerated PARSEC Python path.";

    module.def("build_info", [openmp]() {
        py::dict info;
        info["module"] = "parsec_accelerated_native";
        info["version"] = "0.6.1";
        info["maximum_multipole_order"] = 60;
        info["dtype"] = "float64";
        info["fixed_summation_order"] = true;
        info["implemented_kernels"] = py::make_tuple(
            "build_negative_laplacian_buffers",
            "FusedHamiltonian",
            "ConjugateGradientSolver",
            "MultipoleBoundaryBuilder",
            "SymmetryMultipoleBoundaryBuilder",
            "CALDAEvaluator",
            "RadialGridEvaluator"
        );
        info["openmp_enabled"] = openmp.enabled;
        info["openmp_detected_processors"] = openmp.detected_processors;
        info["openmp_reserved_threads"] = openmp.reserved_threads;
        info["openmp_default_threads"] = openmp.default_threads;
        info["openmp_max_threads"] = openmp.max_threads;
        info["openmp_dynamic"] = openmp.dynamic;
        info["openmp_thread_source"] = openmp.source;
        return info;
    });

    module.def(
        "build_negative_laplacian_buffers",
        &build_negative_laplacian_buffers,
        py::arg("integer_coordinates"),
        py::arg("index_min"),
        py::arg("lookup"),
        py::arg("expansion_order"),
        py::arg("spacing"),
        R"pbdoc(
Build canonical CSR buffers for the compressed-grid negative Laplacian.

The coordinate-to-row lookup determines whether every signed axial stencil
neighbor is active.  Missing neighbors implement homogeneous Dirichlet orbital
boundary values and are omitted without renormalizing the stencil.
)pbdoc"
    );

    py::class_<FusedHamiltonian>(module, "FusedHamiltonian")
        .def(
            py::init<
                const IndexArray&,
                const IndexArray&,
                const FloatArray&,
                const IndexArray&,
                const IndexArray&,
                const FloatArray&,
                const FloatArray&,
                const FloatArray&
            >(),
            py::arg("a_indptr"),
            py::arg("a_indices"),
            py::arg("a_data"),
            py::arg("b_indptr"),
            py::arg("b_indices"),
            py::arg("b_data"),
            py::arg("signs"),
            py::arg("local_potential"),
            R"pbdoc(
Cache a CSR negative Laplacian and CSC KB projector factorization.

All buffers are copied once into native-owned float64/int64 storage.  apply()
releases the Python GIL and evaluates A@Q + V*Q + B*(signs*(B.T@Q)) with
deterministic per-output summation order.
)pbdoc"
        )
        .def(
            "update_local",
            &FusedHamiltonian::update_local,
            py::arg("local_potential"),
            "Replace the cached diagonal effective potential."
        )
        .def(
            "apply",
            &FusedHamiltonian::apply,
            py::arg("vectors"),
            "Apply the fused Hamiltonian to one vector or a column block."
        )
        .def("__matmul__", &FusedHamiltonian::apply, py::is_operator())
        .def_property_readonly("shape", &FusedHamiltonian::shape)
        .def_property_readonly("size", &FusedHamiltonian::size)
        .def_property_readonly(
            "projector_count",
            &FusedHamiltonian::projector_count
        )
        .def_property_readonly("dtype", [](const FusedHamiltonian&) {
            return py::dtype::of<double>();
        });

    py::class_<ConjugateGradientSolver>(
        module,
        "ConjugateGradientSolver"
    )
        .def(
            py::init<const IndexArray&, const IndexArray&, const FloatArray&>(),
            py::arg("indptr"),
            py::arg("indices"),
            py::arg("data"),
            R"pbdoc(
Cache one canonical float64 CSR matrix for repeated CG Poisson solves.

The solve recurrence matches the native-Python Hartree implementation's warm
start, tolerance, matrix-vector-product budget, breakdown rule, and explicit
final residual evaluation.  Native work releases the Python GIL.
)pbdoc"
        )
        .def(
            "solve",
            &ConjugateGradientSolver::solve,
            py::arg("right_hand_side"),
            py::arg("initial"),
            py::arg("relative_tolerance"),
            py::arg("absolute_tolerance"),
            py::arg("max_iterations")
        )
        .def_property_readonly("shape", &ConjugateGradientSolver::shape)
        .def_property_readonly("size", &ConjugateGradientSolver::size)
        .def_property_readonly(
            "worker_count",
            &ConjugateGradientSolver::worker_count
        )
        .def_property_readonly(
            "storage_mode",
            &ConjugateGradientSolver::storage_mode
        )
        .def_property_readonly(
            "coefficient_palette_size",
            &ConjugateGradientSolver::coefficient_palette_size
        )
        .def_property_readonly("dtype", [](const ConjugateGradientSolver&) {
            return py::dtype::of<double>();
        });

    py::class_<MultipoleBoundaryBuilder>(
        module,
        "MultipoleBoundaryBuilder"
    )
        .def(
            py::init<
                const IndexArray&,
                const FloatArray&,
                const IndexArray&,
                const IndexArray&,
                const FloatArray&,
                int,
                double,
                int
            >(),
            py::arg("integer_coordinates"),
            py::arg("coordinates"),
            py::arg("index_min"),
            py::arg("lookup"),
            py::arg("shift"),
            py::arg("expansion_order"),
            py::arg("spacing"),
            py::arg("multipole_order"),
            R"pbdoc(
Cache an isolated spherical grid's multipole and exterior-stencil geometry.

build(density) forms normalized complex moments through l=9 and folds their
Rydberg-unit boundary potential into 8*pi*density.  The per-SCF operation is
float64 C++/OpenMP and releases the Python GIL.
)pbdoc"
        )
        .def(
            py::init<
                int,
                double,
                const IndexArray&,
                const py::array_t<
                    std::complex<double>,
                    py::array::c_style | py::array::forcecast
                >&,
                const IndexArray&,
                const FloatArray&,
                const FloatArray&,
                const FloatArray&,
                const FloatArray&,
                const FloatArray&,
                const FloatArray&
            >(),
            py::arg("multipole_order"),
            py::arg("volume_element"),
            py::arg("multiplicities"),
            py::arg("moment_coefficients"),
            py::arg("boundary_indptr"),
            py::arg("boundary_operator_coefficient"),
            py::arg("boundary_radius"),
            py::arg("boundary_cosine"),
            py::arg("boundary_sine"),
            py::arg("boundary_phase_real"),
            py::arg("boundary_phase_imag"),
            R"pbdoc(
Restore a symmetry multipole/RHS builder from a validated persistent cache.
)pbdoc"
        )
        .def(
            "build",
            &MultipoleBoundaryBuilder::build,
            py::arg("density")
        )
        .def(
            "configure_symmetry",
            &MultipoleBoundaryBuilder::configure_symmetry,
            py::arg("representative_rows"),
            py::arg("full_to_wedge"),
            py::arg("multiplicities"),
            R"pbdoc(
Precompute exact orbit-summed multipole coefficients for invariant densities.
)pbdoc"
        )
        .def(
            "build_reduced",
            &MultipoleBoundaryBuilder::build_reduced,
            py::arg("wedge_density"),
            R"pbdoc(
Build multipoles and the normalized symmetry-wedge Poisson right-hand side.
)pbdoc"
        )
        .def(
            "export_symmetry_cache",
            &MultipoleBoundaryBuilder::export_symmetry_cache,
            R"pbdoc(
Export the exact wedge multipole and exterior-boundary geometry buffers.
)pbdoc"
        )
        .def_property_readonly("size", &MultipoleBoundaryBuilder::size)
        .def("export_full_geometry", &MultipoleBoundaryBuilder::export_full_geometry,
             "Copy the exact source and exterior-stencil geometry for GPU evaluation.")
        .def_property_readonly(
            "boundary_term_count",
            &MultipoleBoundaryBuilder::boundary_term_count
        )
        .def_property_readonly(
            "multipole_order",
            &MultipoleBoundaryBuilder::multipole_order
        )
        .def_property_readonly(
            "symmetry_wedge_size",
            &MultipoleBoundaryBuilder::symmetry_wedge_size
        )
        .def_property_readonly("dtype", [](const MultipoleBoundaryBuilder&) {
            return py::dtype::of<double>();
        });

    py::class_<CALDAEvaluator>(module, "CALDAEvaluator")
        .def(
            py::init<const FloatArray&, double>(),
            py::arg("core_density"),
            py::arg("volume_element"),
            "Cache frozen core density for repeated float64 CA/PZ-LDA calls."
        )
        .def(
            py::init<const FloatArray&, double, const IndexArray&>(),
            py::arg("core_density"),
            py::arg("volume_element"),
            py::arg("integration_weights"),
            "Cache wedge core density and exact orbit quadrature weights."
        )
        .def("evaluate", &CALDAEvaluator::evaluate, py::arg("valence_density"))
        .def_property_readonly("size", &CALDAEvaluator::size)
        .def_property_readonly("dtype", [](const CALDAEvaluator&) {
            return py::dtype::of<double>();
        });

    module.def("reduce_sector_csr", &parsec_accelerated_native::reduce_sector_csr,
        "Project canonical CSR representative rows with bounded thread scratch.");

    module.def(
        "build_sector_stencil",
        &parsec_accelerated_native::build_sector_stencil,
        py::arg("integer_coordinates"),
        py::arg("index_min"),
        py::arg("lookup"),
        py::arg("expansion_order"),
        py::arg("spacing"),
        py::arg("representatives"),
        py::arg("full_to_orbit"),
        py::arg("orbit_to_sector"),
        py::arg("multiplicities"),
        py::arg("phases"),
        py::arg("threads") = 0,
        R"pbdoc(
Build the slot-major stencil of one symmetry sector from the grid lookup.

Only the representative rows of the sector are visited and no full-grid matrix
is formed.  The neighbors, coefficient codes and palette equal those obtained
by reducing the full-grid CSR with reduce_sector_csr and packing the result.

The rows are taken in blocks of ``block_rows`` (returned) by ``threads`` OpenMP
threads, 0 meaning the default team; a sector of fewer rows than one block runs
on the calling thread.  The arrays do not depend on either number.
)pbdoc"
    );

    py::class_<RadialGridEvaluator>(module, "RadialGridEvaluator")
        .def(
            py::init<const FloatArray&>(),
            py::arg("coordinates"),
            "Cache active-grid Cartesian coordinates for atom-radial kernels."
        )
        .def(
            "local_potential",
            &RadialGridEvaluator::local_potential,
            py::arg("atom_position"),
            py::arg("radii"),
            py::arg("values"),
            py::arg("ionic_charge"),
            py::arg("spline_knots"),
            py::arg("spline_values"),
            py::arg("spline_second_derivatives")
        )
        .def(
            "density",
            &RadialGridEvaluator::density,
            py::arg("atom_position"),
            py::arg("radii"),
            py::arg("values"),
            py::arg("spline_knots"),
            py::arg("spline_values"),
            py::arg("spline_second_derivatives")
        )
        .def(
            "projector_channel",
            &RadialGridEvaluator::projector_channel,
            py::arg("atom_position"),
            py::arg("radii"),
            py::arg("radial_projector"),
            py::arg("support_radius"),
            py::arg("angular_momentum"),
            py::arg("square_root_volume"),
            py::arg("spline_knots"),
            py::arg("spline_values"),
            py::arg("spline_second_derivatives")
        )
        .def_property_readonly("size", &RadialGridEvaluator::size)
        .def_property_readonly("dtype", [](const RadialGridEvaluator&) {
            return py::dtype::of<double>();
        });

}
