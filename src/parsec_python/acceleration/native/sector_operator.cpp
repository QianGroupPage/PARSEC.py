#include "sector_operator.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace parsec_accelerated_native {
namespace {
struct CSRIndex {
    const void* pointer;
    bool wide;
    explicit CSRIndex(const py::array& array) : pointer(array.data()), wide(false) {
        if (array.ndim()!=1 || !(array.flags() & py::array::c_style))
            throw std::invalid_argument("CSR indices must be contiguous vectors");
        wide=array.dtype().is(py::dtype::of<std::int64_t>());
        if (!wide && !array.dtype().is(py::dtype::of<std::int32_t>()))
            throw std::invalid_argument("CSR indices must be int32 or int64");
    }
    std::int64_t operator[](std::int64_t i) const {
        return wide ? static_cast<const std::int64_t*>(pointer)[i]
                    : static_cast<const std::int32_t*>(pointer)[i];
    }
};
}

py::dict reduce_sector_csr(
    const py::array& indptr, const py::array& indices, const FloatArray& data,
    const IndexArray& representatives, const IndexArray& full_to_orbit,
    const IndexArray& orbit_to_sector, const IndexArray& multiplicities,
    const py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>& phases) {
    CSRIndex ptr(indptr), cols(indices);
    const auto n=full_to_orbit.size(), rows=representatives.size(), orbits=multiplicities.size();
    if (data.ndim()!=1 || representatives.ndim()!=1 || full_to_orbit.ndim()!=1 ||
        orbit_to_sector.ndim()!=1 || multiplicities.ndim()!=1 || phases.ndim()!=1 ||
        indptr.size()!=n+1 || phases.size()!=n || orbit_to_sector.size()!=orbits ||
        indices.size()!=data.size() || ptr[0]!=0 || ptr[n]!=data.size() ||
        rows>std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("inconsistent sector/CSR dimensions");
    const auto* to_orbit=full_to_orbit.data();
    const auto* to_sector=orbit_to_sector.data();
    const auto* mult=multiplicities.data();
    const auto* reps=representatives.data();
    const auto* phase=phases.data();
    const double* vals=data.data();
    for (py::ssize_t i=0; i<orbits; ++i)
        if (mult[i]<1 || to_sector[i]<-1 || to_sector[i]>=rows)
            throw std::invalid_argument("invalid orbit-to-sector map");
    for (py::ssize_t i=0; i<n; ++i)
        if (ptr[i]>ptr[i+1] || ptr[i]<0 || ptr[i+1]>data.size() ||
            to_orbit[i]<0 || to_orbit[i]>=orbits || phase[i]<-1 || phase[i]>1)
            throw std::invalid_argument("invalid full-grid/CSR map");
    for (py::ssize_t i=0; i<rows; ++i)
        if (reps[i]<0 || reps[i]>=n || to_sector[to_orbit[reps[i]]]!=i)
            throw std::invalid_argument("invalid representative order");
    for (py::ssize_t i=0; i<indices.size(); ++i)
        if (cols[i]<0 || cols[i]>=n)
            throw std::invalid_argument("CSR column out of bounds");

    py::array_t<std::int64_t> outptr(rows+1);
    auto* rowptr=outptr.mutable_data();
    rowptr[0]=0;
    // A thread owns just one row's entries. Two passes avoid nnz-sized
    // phase, normalization, mask, COO-row and column-orbit temporaries.
    auto assemble = [&](std::int64_t row, std::vector<std::pair<std::int64_t,double>>& entries) {
        entries.clear();
        const auto fullrow=reps[row], orbit=to_orbit[fullrow];
        for (auto k=ptr[fullrow]; k<ptr[fullrow+1]; ++k) {
            const auto col=cols[k], col_orbit=to_orbit[col], target=to_sector[col_orbit];
            if (target>=0) entries.emplace_back(target,
                vals[k]*phase[col]*std::sqrt(static_cast<double>(mult[orbit])/mult[col_orbit]));
        }
        std::stable_sort(entries.begin(), entries.end(),
            [](const auto& a,const auto& b){ return a.first<b.first; });
        std::size_t used=0;
        for (std::size_t i=0; i<entries.size();) {
            const auto col=entries[i].first;
            double value=entries[i++].second;
            while (i<entries.size() && entries[i].first==col) value+=entries[i++].second;
            if (value!=0.0) entries[used++]={col,value};
        }
        entries.resize(used);
    };
    {
        py::gil_scoped_release release;
#pragma omp parallel if(rows>=4096)
        {
            std::vector<std::pair<std::int64_t,double>> entries;
#pragma omp for schedule(static)
            for (std::int64_t row=0; row<rows; ++row) {
                assemble(row,entries);
                rowptr[row+1]=entries.size();
            }
        }
        for (std::int64_t row=0; row<rows; ++row) rowptr[row+1]+=rowptr[row];
    }
    py::array_t<std::int64_t> outcols(rowptr[rows]);
    py::array_t<double> outdata(rowptr[rows]);
    auto* column=outcols.mutable_data();
    auto* value=outdata.mutable_data();
    {
        py::gil_scoped_release release;
#pragma omp parallel if(rows>=4096)
        {
            std::vector<std::pair<std::int64_t,double>> entries;
#pragma omp for schedule(static)
            for (std::int64_t row=0; row<rows; ++row) {
                assemble(row,entries);
                for (std::size_t j=0; j<entries.size(); ++j) {
                    column[rowptr[row]+j]=entries[j].first;
                    value[rowptr[row]+j]=entries[j].second;
                }
            }
        }
    }
    py::dict result;
    result["indptr"]=std::move(outptr);
    result["indices"]=std::move(outcols);
    result["data"]=std::move(outdata);
    return result;
}
}
