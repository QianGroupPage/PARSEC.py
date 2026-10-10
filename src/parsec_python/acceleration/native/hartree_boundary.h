#pragma once

#include "finite_difference.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <complex>
#include <vector>

namespace parsec_accelerated_native {

// Cache the static geometry needed by PARSEC's isolated multipole boundary.
// build() then forms Q_lm and b_eff = 8*pi*rho - A_IB*V_B entirely in
// float64 C++/OpenMP for each SCF density.
class MultipoleBoundaryBuilder {
public:
    MultipoleBoundaryBuilder(
        const IndexArray& integer_coordinates,
        const FloatArray& coordinates,
        const IndexArray& index_min,
        const IndexArray& lookup,
        const FloatArray& shift,
        int expansion_order,
        double spacing,
        int multipole_order
    );
    MultipoleBoundaryBuilder(
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
    );

    py::dict build(const FloatArray& density) const;

    // Precompute orbit-summed multipole coefficients for an exact scalar
    // symmetry.  build_reduced() subsequently accepts one physical density
    // value per orbit and returns the normalized wedge RHS U.T b.
    void configure_symmetry(
        const IndexArray& representative_rows,
        const IndexArray& full_to_wedge,
        const IndexArray& multiplicities
    );
    py::dict build_reduced(const FloatArray& wedge_density) const;
    py::dict export_symmetry_cache() const;
    py::dict export_full_geometry() const;

    std::size_t size() const noexcept;
    std::size_t boundary_term_count() const noexcept;
    int multipole_order() const noexcept;
    std::size_t symmetry_wedge_size() const noexcept;

private:
    std::size_t point_count_ = 0;
    int multipole_order_ = 0;
    // Arrays over (l, m) hold stride_*stride_ entries, stride_ = order+1.
    int stride_ = 1;
    std::size_t storage_ = 1;
    double volume_element_ = 0.0;

    // Angular coordinates of the active source points.  Caching these avoids
    // square roots and divisions in every SCF multipole construction.
    std::vector<double> source_radius_;
    std::vector<double> source_cosine_;
    std::vector<double> source_sine_;
    std::vector<double> source_phase_real_;
    std::vector<double> source_phase_imag_;

    // Exterior stencil entries are grouped by interior row, so different
    // OpenMP threads update disjoint RHS entries without atomics.
    std::vector<std::int64_t> boundary_indptr_;
    std::vector<double> boundary_operator_coefficient_;
    std::vector<double> boundary_radius_;
    std::vector<double> boundary_cosine_;
    std::vector<double> boundary_sine_;
    std::vector<double> boundary_phase_real_;
    std::vector<double> boundary_phase_imag_;

    // Normalized complex Y_lm prefactors, stored at l*stride_+m for
    // 0<=m<=l<=order.
    std::vector<double> normalization_;

    // Optional exact-orbit data.  Coefficient (w,lm) is the full sum of
    // h^3 r^l conj(Y_lm) over every full-grid row in orbit w.  Precomputing
    // it moves the expensive angular recurrence out of all SCF iterations.
    std::vector<std::int64_t> symmetry_representative_rows_;
    std::vector<std::int64_t> symmetry_multiplicities_;
    std::size_t symmetry_angular_count_ = 0;
    std::vector<std::complex<double>> symmetry_moment_coefficients_;
};

}  // namespace parsec_accelerated_native
