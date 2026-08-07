#include "pseudo_diag_omp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

namespace py = pybind11;

struct SplineTable {
    std::vector<double> xi;
    std::vector<double> z;
    std::vector<double> c;
    std::vector<double> d;
};

struct SpeciesData {
    std::string typ;
    std::vector<double> coords;
    std::size_t natoms = 0;
    SplineTable charge;
    SplineTable pot_s;
    SplineTable hartree;
};

std::vector<double> copy_1d_array(const py::handle& obj) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!array || array.ndim() != 1) {
        throw std::runtime_error("Expected a 1D float64 NumPy array.");
    }

    const auto view = array.unchecked<1>();
    std::vector<double> values(static_cast<std::size_t>(view.shape(0)));
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
        values[static_cast<std::size_t>(index)] = view(index);
    }
    return values;
}

std::vector<double> copy_2d_array(const py::handle& obj, std::size_t& rows_out) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!array || array.ndim() != 2 || array.shape(1) != 3) {
        throw std::runtime_error("Expected a 2D float64 NumPy array with shape (n, 3).");
    }

    rows_out = static_cast<std::size_t>(array.shape(0));
    const auto view = array.unchecked<2>();
    std::vector<double> values(static_cast<std::size_t>(array.shape(0) * array.shape(1)));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        for (py::ssize_t col = 0; col < array.shape(1); ++col) {
            values[static_cast<std::size_t>(row * 3 + col)] = view(row, col);
        }
    }
    return values;
}

SplineTable parse_spline(
    const py::dict& entry,
    const char* x_key,
    const char* z_key,
    const char* c_key,
    const char* d_key
) {
    SplineTable table;
    table.xi = copy_1d_array(entry[x_key]);
    table.z = copy_1d_array(entry[z_key]);
    table.c = copy_1d_array(entry[c_key]);
    table.d = copy_1d_array(entry[d_key]);
    if (table.xi.size() < 2) {
        throw std::runtime_error("Spline input must contain at least two xi points.");
    }
    if (table.z.size() != table.xi.size()) {
        throw std::runtime_error("Spline z coefficients must match xi length.");
    }
    if (table.c.size() + 1 != table.xi.size() || table.d.size() + 1 != table.xi.size()) {
        throw std::runtime_error("Spline c/d coefficients must have length len(xi) - 1.");
    }
    return table;
}

SpeciesData parse_species_entry(const py::handle& obj) {
    py::dict entry = py::cast<py::dict>(obj);
    SpeciesData species;
    species.typ = py::cast<std::string>(entry["typ"]);
    species.coords = copy_2d_array(entry["coords"], species.natoms);
    species.charge = parse_spline(entry, "x_charge", "z_charge", "c_charge", "d_charge");
    species.pot_s = parse_spline(entry, "x_pot_s", "z_pot_s", "c_pot_s", "d_pot_s");
    species.hartree = parse_spline(entry, "x_hartree", "z_hartree", "c_hartree", "d_hartree");
    return species;
}

double spline_eval(const SplineTable& spline, double x, int& j_in) {
    const int n = static_cast<int>(spline.xi.size());
    if (j_in < 0 || j_in >= n - 1) {
        j_in = 0;
    }

    if (x < spline.xi.front()) {
        x = spline.xi.front();
        j_in = 0;
    } else if (x > spline.xi.back()) {
        x = spline.xi.back();
        j_in = n - 2;
    }

    int j_out = j_in;
    if (!(spline.xi[static_cast<std::size_t>(j_in)] <= x &&
          x <= spline.xi[static_cast<std::size_t>(j_in + 1)])) {
        int ind_low = 0;
        int ind_high = n - 1;
        if (spline.xi.front() <= x && x <= spline.xi.back()) {
            while (ind_high - ind_low > 1) {
                const int ind_middle = (ind_high + ind_low) / 2;
                const double val_middle = spline.xi[static_cast<std::size_t>(ind_middle)];
                if (x < val_middle) {
                    ind_high = ind_middle;
                } else {
                    ind_low = ind_middle;
                }
            }
            j_out = ind_low;
        } else {
            throw std::runtime_error("SPLINE ERROR [ in binary search ]");
        }
    }

    const double t1 = spline.xi[static_cast<std::size_t>(j_out + 1)] - x;
    const double t2 = x - spline.xi[static_cast<std::size_t>(j_out)];
    const double h = spline.xi[static_cast<std::size_t>(j_out + 1)] - spline.xi[static_cast<std::size_t>(j_out)];

    const double y =
        t1 * (spline.z[static_cast<std::size_t>(j_out)] * t1 * t1 / h + spline.c[static_cast<std::size_t>(j_out)]) +
        t2 * (spline.z[static_cast<std::size_t>(j_out + 1)] * t2 * t2 / h + spline.d[static_cast<std::size_t>(j_out)]);

    j_in = j_out + 1;
    return y;
}

}  // namespace

namespace rsdft_native {

py::object pseudo_diag_omp(
    const py::dict& domain,
    const py::list& species,
    double z_sum,
    bool return_info,
    bool build_hpot
) {
    if (!domain.contains("nx") || !domain.contains("ny") || !domain.contains("nz") || !domain.contains("h")) {
        throw std::runtime_error(
            "pseudo_diag_omp expected a domain dict with keys: nx, ny, nz, h."
        );
    }

    const int nx = py::cast<int>(domain["nx"]);
    const int ny = py::cast<int>(domain["ny"]);
    const int nz = py::cast<int>(domain["nz"]);
    const double h = py::cast<double>(domain["h"]);
    const double rad = py::cast<double>(domain["radius"]);
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error("Domain dimensions must be positive.");
    }

    std::vector<SpeciesData> species_data;
    species_data.reserve(static_cast<std::size_t>(py::len(species)));
    for (py::handle entry : species) {
        species_data.push_back(parse_species_entry(entry));
    }

    const py::ssize_t ndim = static_cast<py::ssize_t>(nx) * static_cast<py::ssize_t>(ny) * static_cast<py::ssize_t>(nz);
    py::array_t<double> rho0(ndim);
    py::array_t<double> hpot0(ndim);
    py::array_t<double> pot(ndim);

    auto* rho_ptr = static_cast<double*>(rho0.mutable_data());
    auto* hpot_ptr = static_cast<double*>(hpot0.mutable_data());
    auto* pot_ptr = static_cast<double*>(pot.mutable_data());
    std::fill(rho_ptr, rho_ptr + ndim, 0.0);
    std::fill(hpot_ptr, hpot_ptr + ndim, 0.0);
    std::fill(pot_ptr, pot_ptr + ndim, 0.0);

    std::vector<double> x_grid(static_cast<std::size_t>(nx));
    std::vector<double> y_grid(static_cast<std::size_t>(ny));
    std::vector<double> z_grid(static_cast<std::size_t>(nz));
    for (int index = 0; index < nx; ++index) {
        x_grid[static_cast<std::size_t>(index)] = static_cast<double>(index) * h - rad;
    }
    for (int index = 0; index < ny; ++index) {
        y_grid[static_cast<std::size_t>(index)] = static_cast<double>(index) * h - rad;
    }
    for (int index = 0; index < nz; ++index) {
        z_grid[static_cast<std::size_t>(index)] = static_cast<double>(index) * h - rad;
    }

    const double rho_scale = (h * h * h) / (4.0 * std::acos(-1.0));

    {
        py::gil_scoped_release release;

        for (const auto& entry : species_data) {
            for (std::size_t atom_index = 0; atom_index < entry.natoms; ++atom_index) {
                const double atom_x = entry.coords[atom_index * 3];
                const double atom_y = entry.coords[atom_index * 3 + 1];
                const double atom_z = entry.coords[atom_index * 3 + 2];

                std::vector<double> x_shift_sq(static_cast<std::size_t>(nx));
                std::vector<double> y_shift_sq(static_cast<std::size_t>(ny));
                std::vector<double> z_shift_sq(static_cast<std::size_t>(nz));
                for (int index = 0; index < nx; ++index) {
                    const double dx = x_grid[static_cast<std::size_t>(index)] - atom_x;
                    x_shift_sq[static_cast<std::size_t>(index)] = dx * dx;
                }
                for (int index = 0; index < ny; ++index) {
                    const double dy = y_grid[static_cast<std::size_t>(index)] - atom_y;
                    y_shift_sq[static_cast<std::size_t>(index)] = dy * dy;
                }
                for (int index = 0; index < nz; ++index) {
                    const double dz = z_grid[static_cast<std::size_t>(index)] - atom_z;
                    z_shift_sq[static_cast<std::size_t>(index)] = dz * dz;
                }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int row_index = 0; row_index < ny * nz; ++row_index) {
                    const int k_idx = row_index / ny;
                    const int j_idx = row_index - k_idx * ny;
                    const double z_term = z_shift_sq[static_cast<std::size_t>(k_idx)];
                    const double y_term = y_shift_sq[static_cast<std::size_t>(j_idx)];
                    const std::size_t row_offset =
                        static_cast<std::size_t>(nx) *
                        (static_cast<std::size_t>(j_idx) + static_cast<std::size_t>(k_idx) * static_cast<std::size_t>(ny));

                    int j_charge = 0;
                    int j_pot = 0;
                    int j_hartree = 0;
                    for (int i_idx = 0; i_idx < nx; ++i_idx) {
                        const double radius =
                            std::sqrt(x_shift_sq[static_cast<std::size_t>(i_idx)] + y_term + z_term);
                        const std::size_t linear_index = row_offset + static_cast<std::size_t>(i_idx);

                        pot_ptr[linear_index] += spline_eval(entry.pot_s, radius, j_pot);

                        double rrho = spline_eval(entry.charge, radius, j_charge) * rho_scale;
                        if (rrho < 0.0) {
                            rrho = 0.0;
                        }
                        rho_ptr[linear_index] += rrho;

                        if (build_hpot) {
                            hpot_ptr[linear_index] += spline_eval(entry.hartree, radius, j_hartree);
                        }
                    }
                }
            }
        }
    }

    double rho_sum = 0.0;
    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = -std::numeric_limits<double>::infinity();
    bool any_nonzero = false;
    for (py::ssize_t index = 0; index < ndim; ++index) {
        const double value = rho_ptr[index];
        rho_sum += value;
        rho_min = std::min(rho_min, value);
        rho_max = std::max(rho_max, value);
        any_nonzero = any_nonzero || value != 0.0;
    }

    py::print(
        "[atomic] rho0 sum =",
        py::float_(rho_sum),
        " min =",
        py::float_(rho_min),
        " max =",
        py::float_(rho_max),
        " any_nonzero =",
        py::bool_(any_nonzero)
    );

    if (rho_sum == 0.0) {
        throw std::runtime_error("pseudo_diag_omp produced zero initial density; cannot normalize rho0.");
    }

    const double electron_count_initial = rho_sum;
    for (py::ssize_t index = 0; index < ndim; ++index) {
        rho_ptr[index] = z_sum * rho_ptr[index] / rho_sum;
    }

    double electron_count_norm = 0.0;
    for (py::ssize_t index = 0; index < ndim; ++index) {
        electron_count_norm += rho_ptr[index];
    }
    py::print("[atomic] rho0 sum =", py::float_(electron_count_norm));

    py::dict diag_info;
    diag_info["electron_count_initial"] = electron_count_initial;
    diag_info["electron_count_normalized"] = electron_count_norm;
    diag_info["electron_target"] = z_sum;

    if (return_info) {
        return py::make_tuple(rho0, hpot0, pot, diag_info);
    }
    return py::make_tuple(rho0, hpot0, pot);
}

}  // namespace rsdft_native
