#pragma once

#include "finite_difference.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace parsec_accelerated_native {

/**
 * Unpreconditioned conjugate-gradient solver for one immutable CSR matrix.
 *
 * The matrix buffers are copied into native-owned storage at construction.
 * solve() then releases the Python GIL for every matrix-vector product and
 * vector update.  CSR rows and fixed-size scalar-reduction blocks retain a
 * deterministic summation order, independent of the OpenMP worker count.
 */
class ConjugateGradientSolver {
public:
    ConjugateGradientSolver(
        const IndexArray& indptr,
        const IndexArray& indices,
        const FloatArray& data
    );

    py::dict solve(
        const FloatArray& right_hand_side,
        const FloatArray& initial,
        double relative_tolerance,
        double absolute_tolerance,
        std::int64_t max_iterations
    ) const;

    [[nodiscard]] std::pair<std::int64_t, std::int64_t> shape() const noexcept;
    [[nodiscard]] std::int64_t size() const noexcept;
    [[nodiscard]] int worker_count() const noexcept;
    [[nodiscard]] std::string storage_mode() const;
    [[nodiscard]] std::size_t coefficient_palette_size() const noexcept;

private:
    void matvec(const double* input, double* output) const noexcept;
    double matvec_and_dot(
        const double* input,
        double* output,
        std::vector<double>& partial
    ) const noexcept;

    std::int64_t size_ = 0;
    std::vector<std::int64_t> indptr_;
    std::vector<std::int64_t> indices_;
    std::vector<double> data_;
    bool compact_storage_ = false;
    std::vector<std::int32_t> compact_indices_;
    std::vector<std::uint8_t> coefficient_codes_;
    std::vector<double> coefficient_palette_;

};

}  // namespace parsec_accelerated_native
