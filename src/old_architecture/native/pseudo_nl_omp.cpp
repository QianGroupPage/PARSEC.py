#include "pseudo_nl_omp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
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
    double xint = 0.0;
    double rzero = 0.0;
    SplineTable wav_p;
    SplineTable pot_ps;
};

struct Triplets {
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> data;
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
    species.xint = py::cast<double>(entry["xint"]);
    species.rzero = py::cast<double>(entry["rzero"]);
    species.wav_p = parse_spline(entry, "xi_wfn_p", "z_wfn_p", "c_wfn_p", "d_wfn_p");
    species.pot_ps = parse_spline(entry, "xi_pot_ps", "z_pot_ps", "c_pot_ps", "d_pot_ps");
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

void append_triplets(Triplets& dst, Triplets&& src) {
    dst.rows.insert(dst.rows.end(), src.rows.begin(), src.rows.end());
    dst.cols.insert(dst.cols.end(), src.cols.begin(), src.cols.end());
    dst.data.insert(dst.data.end(), src.data.begin(), src.data.end());
}

Triplets build_atom_triplets(
    const SpeciesData& species,
    std::size_t atom_index,
    int nx,
    int ny,
    int nz,
    double h,
    double rad
) {
    Triplets out;

    const double atom_x = species.coords[atom_index * 3];
    const double atom_y = species.coords[atom_index * 3 + 1];
    const double atom_z = species.coords[atom_index * 3 + 2];

    const int i0 = static_cast<int>(std::llround((atom_x + rad) / h + 1.0));
    const int j0 = static_cast<int>(std::llround((atom_y + rad) / h + 1.0));
    const int k0 = static_cast<int>(std::llround((atom_z + rad) / h + 1.0));
    const int span = static_cast<int>(std::llround(species.rzero / h));

    const int i_min = std::max(1, i0 - span);
    const int i_max = std::min(nx, i0 + span);
    const int j_min = std::max(1, j0 - span);
    const int j_max = std::min(ny, j0 + span);
    const int k_min = std::max(1, k0 - span);
    const int k_max = std::min(nz, k0 + span);

    if (i_min > i_max || j_min > j_max || k_min > k_max) {
        return out;
    }

    const std::size_t max_points = static_cast<std::size_t>(i_max - i_min + 1) *
                                   static_cast<std::size_t>(j_max - j_min + 1) *
                                   static_cast<std::size_t>(k_max - k_min + 1);

    std::vector<std::int64_t> nn;
    std::vector<double> xx;
    std::vector<double> yy;
    std::vector<double> zz;
    std::vector<double> dd;
    std::vector<double> vspp;
    std::vector<double> wavpp;
    nn.reserve(max_points);
    xx.reserve(max_points);
    yy.reserve(max_points);
    zz.reserve(max_points);
    dd.reserve(max_points);
    vspp.reserve(max_points);
    wavpp.reserve(max_points);

    for (int k = k_min; k <= k_max; ++k) {
        const double zzz = static_cast<double>(k - 1) * h - rad - atom_z;
        for (int j = j_min; j <= j_max; ++j) {
            const double yyy = static_cast<double>(j - 1) * h - rad - atom_y;
            int j_p_ps = 1;
            int j_wfn = 1;
            for (int i = i_min; i <= i_max; ++i) {
                const double xxx = static_cast<double>(i - 1) * h - rad - atom_x;
                const double dd1 = std::sqrt(xxx * xxx + yyy * yyy + zzz * zzz);
                if (dd1 <= 0.0 || dd1 >= species.rzero) {
                    continue;
                }

                nn.push_back(
                    static_cast<std::int64_t>(i - 1) +
                    static_cast<std::int64_t>(nx) *
                        (static_cast<std::int64_t>(j - 1) + static_cast<std::int64_t>(k - 1) * static_cast<std::int64_t>(ny))
                );
                xx.push_back(xxx);
                yy.push_back(yyy);
                zz.push_back(zzz);
                dd.push_back(dd1);
                vspp.push_back(spline_eval(species.pot_ps, dd1, j_p_ps));
                wavpp.push_back(spline_eval(species.wav_p, dd1, j_wfn));
            }
        }
    }

    const std::size_t count = nn.size();
    if (count == 0) {
        return out;
    }

    std::vector<double> ulmspx(count);
    std::vector<double> ulmspy(count);
    std::vector<double> ulmspz(count);
    for (std::size_t index = 0; index < count; ++index) {
        const double fac = wavpp[index] * vspp[index];
        ulmspx[index] = xx[index] / dd[index] * fac;
        ulmspy[index] = yy[index] / dd[index] * fac;
        ulmspz[index] = zz[index] / dd[index] * fac;
    }

    out.rows.reserve(count * count);
    out.cols.reserve(count * count);
    out.data.reserve(count * count);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t col = 0; col < count; ++col) {
            const double value =
                (ulmspx[row] * ulmspx[col] +
                 ulmspy[row] * ulmspy[col] +
                 ulmspz[row] * ulmspz[col]) /
                species.xint;
            out.rows.push_back(nn[row]);
            out.cols.push_back(nn[col]);
            out.data.push_back(value);
        }
    }

    return out;
}

template <typename T>
py::array_t<T> vector_to_array(const std::vector<T>& values) {
    py::array_t<T> array(values.size());
    auto view = array.template mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < view.shape(0); ++index) {
        view(index) = values[static_cast<std::size_t>(index)];
    }
    return array;
}

}  // namespace

namespace rsdft_native {

py::object pseudo_nl_omp(
    const py::dict& domain,
    const py::list& species
) {
    if (!domain.contains("nx") || !domain.contains("ny") || !domain.contains("nz") || !domain.contains("h")) {
        throw std::runtime_error(
            "pseudo_nl_omp expected a domain dict with keys: nx, ny, nz, h."
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

    Triplets global_triplets;
    {
        py::gil_scoped_release release;
        for (const auto& entry : species_data) {
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                Triplets thread_triplets;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (int atom_index = 0; atom_index < static_cast<int>(entry.natoms); ++atom_index) {
                    append_triplets(
                        thread_triplets,
                        build_atom_triplets(entry, static_cast<std::size_t>(atom_index), nx, ny, nz, h, rad)
                    );
                }

#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    append_triplets(global_triplets, std::move(thread_triplets));
                }
            }
        }
    }

    py::dict payload;
    payload["rows"] = vector_to_array(global_triplets.rows);
    payload["cols"] = vector_to_array(global_triplets.cols);
    payload["data"] = vector_to_array(global_triplets.data);
    return payload;
}

}  // namespace rsdft_native
