#include "sector_stencil.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace parsec_accelerated_native {
namespace {

struct SectorEntry {
    std::int64_t target;
    std::int64_t column;
    double value;
};

// Rows of one work item.  A block keeps its own list of the coefficient
// values it met, so the result does not depend on the thread count.
constexpr std::int64_t kBlockRows = 4096;

enum BuildError {
    kNoError = 0,
    kLookupMismatch = 1,
    kRowOutsideGrid = 2,
    kTooManyCoefficients = 3,
};

std::uint64_t bit_pattern(const double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}  // namespace

py::dict build_sector_stencil(
    const IndexArray& integer_coordinates,
    const IndexArray& index_min,
    const IndexArray& lookup,
    const int expansion_order,
    const double spacing,
    const IndexArray& representatives,
    const IndexArray& full_to_orbit,
    const IndexArray& orbit_to_sector,
    const IndexArray& multiplicities,
    const py::array_t<
        std::int8_t,
        py::array::c_style | py::array::forcecast
    >& phases,
    const int threads
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
    if (threads < 0) {
        throw std::invalid_argument("threads cannot be negative");
    }
    const std::int64_t point_count =
        static_cast<std::int64_t>(integer_coordinates.shape(0));
    const std::int64_t rows =
        static_cast<std::int64_t>(representatives.size());
    const std::int64_t orbits =
        static_cast<std::int64_t>(multiplicities.size());
    if (
        representatives.ndim() != 1 || full_to_orbit.ndim() != 1 ||
        orbit_to_sector.ndim() != 1 || multiplicities.ndim() != 1 ||
        phases.ndim() != 1 || full_to_orbit.size() != point_count ||
        phases.size() != point_count || orbit_to_sector.size() != orbits ||
        rows < 1 || rows > std::numeric_limits<std::int32_t>::max()
    ) {
        throw std::invalid_argument("inconsistent sector/grid dimensions");
    }

    // The values of build_negative_laplacian_buffers, from the same weights.
    const auto coefficients =
        centered_second_derivative_coefficients(expansion_order);
    const int width = expansion_order / 2;
    const double inverse_spacing_squared = 1.0 / (spacing * spacing);
    const double diagonal =
        -3.0 * coefficients[static_cast<std::size_t>(width)] *
        inverse_spacing_squared;
    std::vector<double> shell_values(static_cast<std::size_t>(width + 1), 0.0);
    for (int shell = 1; shell <= width; ++shell) {
        shell_values[static_cast<std::size_t>(shell)] =
            -coefficients[static_cast<std::size_t>(width + shell)] *
            inverse_spacing_squared;
    }
    const int maximum_entries = 1 + 6 * width;

    const auto* coordinates = integer_coordinates.data();
    const auto* minima = index_min.data();
    const auto* lookup_data = lookup.data();
    const std::int64_t lookup_shape[3] = {
        static_cast<std::int64_t>(lookup.shape(0)),
        static_cast<std::int64_t>(lookup.shape(1)),
        static_cast<std::int64_t>(lookup.shape(2)),
    };
    const auto* reps = representatives.data();
    const auto* to_orbit = full_to_orbit.data();
    const auto* to_sector = orbit_to_sector.data();
    const auto* mult = multiplicities.data();
    const auto* phase = phases.data();

    for (std::int64_t orbit = 0; orbit < orbits; ++orbit) {
        if (
            mult[orbit] < 1 || to_sector[orbit] < -1 ||
            to_sector[orbit] >= rows
        ) {
            throw std::invalid_argument("invalid orbit-to-sector map");
        }
    }
    for (std::int64_t point = 0; point < point_count; ++point) {
        if (
            to_orbit[point] < 0 || to_orbit[point] >= orbits ||
            phase[point] < -1 || phase[point] > 1
        ) {
            throw std::invalid_argument("invalid full-grid symmetry map");
        }
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        if (
            reps[row] < 0 || reps[row] >= point_count ||
            to_sector[to_orbit[reps[row]]] != row
        ) {
            throw std::invalid_argument("invalid representative order");
        }
    }

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

    // One reduced row: the stencil points of its representative grid point,
    // each sent to the sector column of its orbit.  Sorting by sector column
    // and then by full-grid column, and adding equal sector columns from left
    // to right, is the order in which reduce_sector_csr reads a canonical
    // full-grid CSR row.  A sum that is exactly zero is no entry.
    auto assemble = [&](const std::int64_t row, SectorEntry* entries,
                        int& error) -> int {
        const std::int64_t full_row = reps[row];
        const std::int64_t orbit = to_orbit[full_row];
        const auto base = static_cast<std::size_t>(3 * full_row);
        const std::int64_t point[3] = {
            coordinates[base],
            coordinates[base + 1],
            coordinates[base + 2],
        };
        if (row_for_point(point[0], point[1], point[2]) != full_row) {
            error = kLookupMismatch;
            return 0;
        }
        int count = 0;
        auto add = [&](const std::int64_t column, const double coefficient) {
            const std::int64_t column_orbit = to_orbit[column];
            const std::int64_t target = to_sector[column_orbit];
            if (target < 0) {
                return;
            }
            // sqrt(m/m) is exactly one, so equal orbit sizes need no root.
            const double normalization =
                mult[orbit] == mult[column_orbit]
                    ? 1.0
                    : std::sqrt(
                          static_cast<double>(mult[orbit]) / mult[column_orbit]
                      );
            entries[count].target = target;
            entries[count].column = column;
            entries[count].value =
                coefficient * phase[column] * normalization;
            ++count;
        };
        add(full_row, diagonal);
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
                    error = kRowOutsideGrid;
                    return 0;
                }
                add(
                    neighbor,
                    shell_values[static_cast<std::size_t>(std::abs(signed_shell))]
                );
            }
        }
        for (int position = 1; position < count; ++position) {
            const SectorEntry moved = entries[position];
            int hole = position;
            while (
                hole > 0 &&
                (entries[hole - 1].target > moved.target ||
                 (entries[hole - 1].target == moved.target &&
                  entries[hole - 1].column > moved.column))
            ) {
                entries[hole] = entries[hole - 1];
                --hole;
            }
            entries[hole] = moved;
        }
        int used = 0;
        for (int position = 0; position < count;) {
            const std::int64_t target = entries[position].target;
            double value = entries[position++].value;
            while (position < count && entries[position].target == target) {
                value += entries[position++].value;
            }
            if (value != 0.0) {
                entries[used].target = target;
                entries[used].value = value;
                ++used;
            }
        }
        return used;
    };

    const std::int64_t block_count = (rows + kBlockRows - 1) / kBlockRows;
    py::array_t<std::int32_t> packed_neighbors(
        {static_cast<py::ssize_t>(maximum_entries),
         static_cast<py::ssize_t>(rows)}
    );
    py::array_t<std::uint8_t> packed_codes(
        {static_cast<py::ssize_t>(maximum_entries),
         static_cast<py::ssize_t>(rows)}
    );
    auto* neighbors = packed_neighbors.mutable_data();
    auto* codes = packed_codes.mutable_data();
    std::vector<std::uint8_t> widths(static_cast<std::size_t>(rows), 0);
    std::vector<std::vector<std::uint64_t>> block_values(
        static_cast<std::size_t>(block_count)
    );
    std::vector<int> block_widths(static_cast<std::size_t>(block_count), 0);
    std::vector<double> block_asymmetry(
        static_cast<std::size_t>(block_count), 0.0
    );
    std::vector<double> block_scale(static_cast<std::size_t>(block_count), 0.0);
    std::vector<std::uint64_t> palette;
    int failure = kNoError;
    int slot_count = 0;
    double maximum_asymmetry = 0.0;
    double scale = 1.0;
#ifdef _OPENMP
    const int team = threads > 0 ? threads : omp_get_max_threads();
#endif

    {
        py::gil_scoped_release release;

        // Pass 1: neighbors, and coefficient codes numbered within the block.
#pragma omp parallel num_threads(team) if(rows >= kBlockRows)
        {
            std::vector<SectorEntry> entries(
                static_cast<std::size_t>(maximum_entries)
            );
#pragma omp for schedule(dynamic)
            for (std::int64_t block = 0; block < block_count; ++block) {
                auto& values = block_values[static_cast<std::size_t>(block)];
                const std::int64_t first = block * kBlockRows;
                const std::int64_t last = std::min(rows, first + kBlockRows);
                int widest = 0;
                int error = kNoError;
                for (std::int64_t row = first; row < last && !error; ++row) {
                    const int used = assemble(row, entries.data(), error);
                    if (error) {
                        break;
                    }
                    widths[static_cast<std::size_t>(row)] =
                        static_cast<std::uint8_t>(used);
                    widest = std::max(widest, used);
                    for (int slot = 0; slot < used; ++slot) {
                        const std::uint64_t bits =
                            bit_pattern(entries[slot].value);
                        std::size_t code = 0;
                        while (code < values.size() && values[code] != bits) {
                            ++code;
                        }
                        if (code == values.size()) {
                            if (code == 256) {
                                error = kTooManyCoefficients;
                                break;
                            }
                            values.push_back(bits);
                        }
                        const std::int64_t offset =
                            static_cast<std::int64_t>(slot) * rows + row;
                        neighbors[offset] =
                            static_cast<std::int32_t>(entries[slot].target);
                        codes[offset] = static_cast<std::uint8_t>(code);
                    }
                    for (int slot = used; slot < maximum_entries; ++slot) {
                        const std::int64_t offset =
                            static_cast<std::int64_t>(slot) * rows + row;
                        neighbors[offset] = -1;
                        codes[offset] = 0;
                    }
                }
                block_widths[static_cast<std::size_t>(block)] = widest;
                if (error) {
#pragma omp critical(parsec_sector_stencil_failure)
                    failure = error;
                }
            }
        }

        if (!failure) {
            for (const auto& values : block_values) {
                palette.insert(palette.end(), values.begin(), values.end());
            }
            // Ascending bit patterns, the order of numpy.unique on uint64.
            std::sort(palette.begin(), palette.end());
            palette.erase(
                std::unique(palette.begin(), palette.end()), palette.end()
            );
            if (palette.size() > 256) {
                failure = kTooManyCoefficients;
            }
        }
        if (!failure) {
            for (const int widest : block_widths) {
                slot_count = std::max(slot_count, widest);
            }
            std::vector<double> palette_values(palette.size());
            if (!palette.empty()) {
                std::memcpy(
                    palette_values.data(),
                    palette.data(),
                    palette.size() * sizeof(double)
                );
            }

            // Pass 2: renumber the codes into the common palette, then
            // compare every entry with its transpose.  An entry without a
            // transpose counts with its full magnitude, as in A - A.T.
#pragma omp parallel num_threads(team) if(rows >= kBlockRows)
            {
#pragma omp for schedule(dynamic)
                for (std::int64_t block = 0; block < block_count; ++block) {
                    const auto& values =
                        block_values[static_cast<std::size_t>(block)];
                    std::uint8_t renumbered[256];
                    for (std::size_t code = 0; code < values.size(); ++code) {
                        renumbered[code] = static_cast<std::uint8_t>(
                            std::lower_bound(
                                palette.begin(), palette.end(), values[code]
                            ) - palette.begin()
                        );
                    }
                    const std::int64_t first = block * kBlockRows;
                    const std::int64_t last =
                        std::min(rows, first + kBlockRows);
                    const int widest =
                        block_widths[static_cast<std::size_t>(block)];
                    for (int slot = 0; slot < widest; ++slot) {
                        auto* slot_codes =
                            codes + static_cast<std::int64_t>(slot) * rows;
                        for (std::int64_t row = first; row < last; ++row) {
                            if (slot < widths[static_cast<std::size_t>(row)]) {
                                slot_codes[row] = renumbered[slot_codes[row]];
                            }
                        }
                    }
                }
#pragma omp for schedule(dynamic)
                for (std::int64_t block = 0; block < block_count; ++block) {
                    const std::int64_t first = block * kBlockRows;
                    const std::int64_t last =
                        std::min(rows, first + kBlockRows);
                    double asymmetry = 0.0;
                    double largest = 0.0;
                    for (std::int64_t row = first; row < last; ++row) {
                        const int used = widths[static_cast<std::size_t>(row)];
                        for (int slot = 0; slot < used; ++slot) {
                            const std::int64_t offset =
                                static_cast<std::int64_t>(slot) * rows + row;
                            const std::int64_t column = neighbors[offset];
                            const double value = palette_values[codes[offset]];
                            largest = std::max(largest, std::fabs(value));
                            // Row ``column`` holds its sector columns in
                            // ascending order; find ``row`` among them.
                            int low = 0;
                            int high =
                                widths[static_cast<std::size_t>(column)];
                            while (low < high) {
                                const int middle = (low + high) / 2;
                                if (
                                    neighbors[
                                        static_cast<std::int64_t>(middle) *
                                            rows + column
                                    ] < row
                                ) {
                                    low = middle + 1;
                                } else {
                                    high = middle;
                                }
                            }
                            double transposed = 0.0;
                            if (low < widths[static_cast<std::size_t>(column)]) {
                                const std::int64_t partner =
                                    static_cast<std::int64_t>(low) * rows +
                                    column;
                                if (neighbors[partner] == row) {
                                    transposed = palette_values[codes[partner]];
                                }
                            }
                            asymmetry = std::max(
                                asymmetry, std::fabs(value - transposed)
                            );
                        }
                    }
                    block_asymmetry[static_cast<std::size_t>(block)] = asymmetry;
                    block_scale[static_cast<std::size_t>(block)] = largest;
                }
            }
            for (std::int64_t block = 0; block < block_count; ++block) {
                maximum_asymmetry = std::max(
                    maximum_asymmetry,
                    block_asymmetry[static_cast<std::size_t>(block)]
                );
                scale = std::max(
                    scale, block_scale[static_cast<std::size_t>(block)]
                );
            }
        }
    }

    if (failure == kLookupMismatch) {
        throw std::invalid_argument(
            "lookup does not map integer_coordinates back to their rows"
        );
    }
    if (failure == kRowOutsideGrid) {
        throw std::invalid_argument(
            "lookup contains an active row outside the grid"
        );
    }
    if (failure == kTooManyCoefficients) {
        throw std::invalid_argument(
            "finite-difference operator has more than 256 coefficients"
        );
    }
    if (slot_count < 1) {
        throw std::invalid_argument(
            "finite-difference operator contains no entries"
        );
    }

    py::array_t<double> packed_palette(
        static_cast<py::ssize_t>(palette.size())
    );
    std::memcpy(
        packed_palette.mutable_data(),
        palette.data(),
        palette.size() * sizeof(double)
    );
    py::dict result;
    if (slot_count < maximum_entries) {
        // No row uses the last slots; return the leading ones only.
        py::array_t<std::int32_t> trimmed_neighbors(
            {static_cast<py::ssize_t>(slot_count),
             static_cast<py::ssize_t>(rows)}
        );
        py::array_t<std::uint8_t> trimmed_codes(
            {static_cast<py::ssize_t>(slot_count),
             static_cast<py::ssize_t>(rows)}
        );
        const auto entries_kept =
            static_cast<std::size_t>(slot_count) *
            static_cast<std::size_t>(rows);
        std::memcpy(
            trimmed_neighbors.mutable_data(),
            neighbors,
            entries_kept * sizeof(std::int32_t)
        );
        std::memcpy(trimmed_codes.mutable_data(), codes, entries_kept);
        result["neighbors"] = std::move(trimmed_neighbors);
        result["codes"] = std::move(trimmed_codes);
    } else {
        result["neighbors"] = std::move(packed_neighbors);
        result["codes"] = std::move(packed_codes);
    }
    result["palette"] = std::move(packed_palette);
    result["maximum_asymmetry"] = maximum_asymmetry;
    result["scale"] = scale;
    result["block_rows"] = kBlockRows;
    return result;
}

}  // namespace parsec_accelerated_native
