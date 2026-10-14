"""Bounded-memory FP64 GPU multipoles and isolated-boundary Poisson RHS.

Source harmonics are evaluated in registers and reduced into small fixed
blocks. No grid-by-angular-momentum table is materialized. The native builder
provides the existing exact exterior-stencil geometry once, then is released.
The host return interface preserves the existing reduced CG and SCF algebra.

Three builders share the recurrences.  ``CuPyMultipoleBoundaryBuilder`` works
on the full grid with the kernels PARSEC's boundary has always used here.
``CuPySymmetryMultipoleBoundaryBuilder`` works on the wedge of an
axis-reflection group, where the angular work of a high multipole order
shrinks with the group: it serves the boundary whose order is raised.
``CuPyPointMultipoleBoundaryBuilder`` runs its kernels under the identity
alone, for a raised order where no such group reduces the Hartree problem.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from ..backends.cupy import require_cupy
from ..backends.cupy_compile import compile_cupy_raw
from ..SCF.symmetry_fields import SymmetryScalarField
from ..Symmetry import AxisReflectionReduction, SignedPermutationReduction
from .fast_multipole import FastMultipoleExpansion
from .native_boundary import NativeMultipoleBoundaryBuilder


def auto_gpu_boundary_requested(wedge_rows: int, multipole_order: int) -> bool:
    """Select the measured large-A100 case; retain cheap native cached cases.

    The current native orbit table is capped at 512 MiB. Below that bound it
    avoids repeated harmonics already. Other GPU models retain the native
    default until end-to-end measurements justify enabling this policy there.
    Explicit ``PARSEC_HARTREE_BOUNDARY_BACKEND=cupy`` bypasses this policy.
    """
    angular_count = (multipole_order+1)*(multipole_order+2)//2
    if wedge_rows*angular_count*16 <= 512*1024**2:
        return False
    try:
        cp, _ = require_cupy()
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    except RuntimeError:
        return False
    if isinstance(name, bytes):
        name = name.decode('ascii', errors='replace')
    return 'A100' in str(name)

_SOURCE = r'''
__device__ void phase_step(double& re, double& im, double ur, double ui) {
    double next_re = re*ur-im*ui;
    im = re*ui+im*ur;
    re = next_re;
}
extern "C" __global__ void moments(
    long long n, int order, double volume, const double* rho,
    const double* radius, const double* cosine, const double* sine,
    const double* phase_re, const double* phase_im, const double* norm,
    double* partial, int stride) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    int m=blockIdx.y;
    double r=row<n?radius[row]:1.0;
    double c=row<n?cosine[row]:1.0;
    double s=row<n?sine[row]:0.0;
    double ur=row<n?phase_re[row]:1.0, ui=row<n?phase_im[row]:0.0;
    double weight=row<n?rho[row]*volume:0.0;
    double pr=1.0, pi=0.0, diagonal=1.0, radial=1.0;
    for(int k=1;k<=m;++k) {
        diagonal*=-(2*k-1)*s;
        phase_step(pr,pi,ur,ui);
        radial*=r;
    }
    double previous=0.0, current=diagonal;
    __shared__ double real_warp[8], imag_warp[8];
    for(int l=m;l<=order;++l) {
        if(l==m+1) {previous=diagonal; current=(2*m+1)*c*diagonal;}
        else if(l>m+1) {
            double next=((2*l-1)*c*current-(l+m-1)*previous)/(l-m);
            previous=current; current=next;
        }
        double a=weight*radial*(norm[l*stride+m]*current*pr);
        double b=-weight*radial*(norm[l*stride+m]*current*pi);
        for(int offset=16;offset;offset>>=1) {
            a+=__shfl_down_sync(0xffffffff,a,offset);
            b+=__shfl_down_sync(0xffffffff,b,offset);
        }
        if((threadIdx.x&31)==0) {
            real_warp[threadIdx.x>>5]=a; imag_warp[threadIdx.x>>5]=b;
        }
        __syncthreads();
        if(threadIdx.x==0) {
            a=0.0; b=0.0;
            for(int k=0;k<8;++k) {a+=real_warp[k]; b+=imag_warp[k];}
            partial[((l*stride+m)*2)*gridDim.x+blockIdx.x]=a;
            partial[((l*stride+m)*2+1)*gridDim.x+blockIdx.x]=b;
        }
        __syncthreads();
        radial*=r;
    }
}
__device__ double exterior(double r,double c,double s,double ur,double ui,
                           int order,const double* norm,const double* q,int stride) {
    const double pi_const=3.141592653589793238462643383279502884;
    double pr=1.0,pi=0.0,diagonal=1.0,inverse_m=1.0/r,result=0.0;
    for(int m=0;m<=order;++m) {
        if(m) {diagonal*=-(2*m-1)*s; phase_step(pr,pi,ur,ui); inverse_m/=r;}
        double previous=0.0,current=diagonal,radial=inverse_m;
        for(int l=m;l<=order;++l) {
            if(l==m+1) {previous=diagonal; current=(2*m+1)*c*diagonal;}
            else if(l>m+1) {
                double next=((2*l-1)*c*current-(l+m-1)*previous)/(l-m);
                previous=current; current=next;
            }
            double h=norm[l*stride+m]*current;
            int index=(l*stride+m)*2;
            double positive=q[index]*(h*pr)-q[index+1]*(h*pi);
            result+=(4*pi_const/(2*l+1)*radial)*(m?2*positive:positive);
            radial/=r;
        }
    }
    return 2*result;
}
extern "C" __global__ void boundary_rhs(
    long long n,int order,const double* rho,const long long* indptr,
    const double* coefficient,const double* radius,const double* cosine,
    const double* sine,const double* phase_re,const double* phase_im,
    const double* norm,const double* q,double* rhs,int stride) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    double value=8*3.141592653589793238462643383279502884*rho[row];
    for(long long t=indptr[row];t<indptr[row+1];++t)
        value-=coefficient[t]*exterior(radius[t],cosine[t],sine[t],
                                     phase_re[t],phase_im[t],order,norm,q,stride);
    rhs[row]=value;
}
'''


def _stride_normalization(table, order):
    """Return the native ``Y_lm`` prefactors laid out at ``l*(order+1)+m``.

    Extensions before 0.6.0 export 100 values at ``l*10+m`` whatever the
    order, and 0.6.1 does for orders up to 9, so that the kernels of an
    older Python tree find them where they read; the same values are taken
    from there.  0.6.0 exports ``(order+1)**2`` values at every order.
    """
    stride = order+1
    table = np.asarray(table, dtype=np.float64)
    if table.size == stride*stride:
        return table
    if table.size == 100 and stride <= 10:
        return np.ascontiguousarray(table.reshape(10, 10)[:stride, :stride]).reshape(-1)
    raise RuntimeError("the native geometry exporter does not match the multipole order; rebuild the extension")


def _truncated_potential(expansion, order, points):
    """Potential of the terms up to ``order`` of an expansion at host points."""
    order = min(int(order), expansion.order)
    return FastMultipoleExpansion(order=order, moments={
        key: value for key, value in expansion.moments.items() if key[0] <= order
    }).potential(points)


_KEY_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def _row_keys(table):
    """One 64-bit key per row, mixed from the bit patterns of its columns."""
    # Adding zero gives -0.0 the bits of 0.0, which compares equal to it.
    bits = np.ascontiguousarray(table+0.0).view(np.uint64)
    key = bits[:, 0].copy()
    for column in range(1, bits.shape[1]):
        key ^= key >> np.uint64(29)
        key *= _KEY_MULTIPLIER
        key += bits[:, column]
    key ^= key >> np.uint64(32)
    return key


def unique_rows(table):
    """``np.unique(table, axis=0, return_inverse=True)`` for finite float rows.

    Sorting whole rows is most of the host set-up of a builder that keeps
    one value per exterior point.  One key per row is sorted here instead,
    the rows of each key are then compared, and the unique rows are put in
    the order of the row sort.  Rows that share a key without being equal
    send the table through the row sort itself.
    """
    table = np.ascontiguousarray(table, dtype=np.float64)
    _, first, inverse = np.unique(_row_keys(table), return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    rows = table[first]
    if not np.array_equal(rows[inverse], table):
        rows, inverse = np.unique(table, axis=0, return_inverse=True)
        return rows, inverse.reshape(-1)
    order = np.lexsort(rows.T[::-1])
    rank = np.empty(order.size, dtype=np.intp)
    rank[order] = np.arange(order.size)
    return rows[order], rank[inverse]


def _check_points(count, radius, tail=None, nearest=None, seed=0):
    """Indices of ``count`` exterior points where the boundary error is looked for.

    The error of the boundary values is largest close to the sphere and
    close to the outermost atoms, and its maximum over all points is not
    where a single measure of that puts it.  Equal shares therefore take
    the points of the smallest ``radius``, of the largest ``|tail|`` and
    of the smallest distance ``nearest`` to an atom, where those are given,
    each share from the points the earlier ones left; a last share is
    random.  ``count`` at or above the number of points takes them all.
    """
    total = int(radius.size)
    count = min(int(count), total)
    scores = [np.asarray(radius)]
    if tail is not None:
        scores.append(-np.abs(np.asarray(tail)))
    if nearest is not None:
        scores.append(np.asarray(nearest))
    share = count//(len(scores)+1)
    free = np.ones(total, dtype=bool)
    picked = []
    for score in scores:
        order = np.argsort(score, kind="stable")
        taken = order[free[order]][:share]
        free[taken] = False
        picked.append(taken)
    random = np.random.default_rng(seed).choice(
        np.flatnonzero(free), size=count-(total-int(free.sum())), replace=False)
    return np.concatenate((*picked, np.sort(random))).astype(np.int64)


def _cartesian(radius, cosine, sine, phase_re, phase_im):
    """Device coordinates of points held as the angular geometry of the kernels."""
    cp, _ = require_cupy()
    return cp.stack((radius*sine*phase_re, radius*sine*phase_im, radius*cosine), axis=1)


class CuPyMultipoleBoundaryBuilder:
    """Full-grid isolated boundary with O(N) storage; angular arrays follow the order."""

    def __init__(self, grid, multipole_order=9, device_id=None):
        cp, _ = require_cupy()
        native = NativeMultipoleBoundaryBuilder(grid, multipole_order)
        if not hasattr(native._native_builder, "export_full_geometry"):
            raise RuntimeError("GPU boundary requires the rebuilt native geometry exporter")
        self.grid, self.order = grid, native.order
        self.boundary_term_count = native.boundary_term_count
        # Geometry, work arrays and compiled kernels live on one device: the
        # current one, or the device of this process the caller names.
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        payload = dict(native._native_builder.export_full_geometry())
        self.stride = self.order+1
        payload['normalization'] = _stride_normalization(payload['normalization'], self.order)
        with cp.cuda.Device(self.device_id):
            self.geometry = {name: cp.asarray(array) for name, array in payload.items()}
            # No retained host copies of the native geometry or exported buffers.
            del payload, native
            self.blocks = (grid.size+255)//256
            self.partial = cp.zeros((2*self.stride*self.stride, self.blocks), dtype=cp.float64)
            self.density = cp.empty(grid.size, dtype=cp.float64)
            self.rhs = cp.empty_like(self.density)
            self.moment_kernel = cp.RawKernel(_SOURCE, "moments", options=("--std=c++11",))
            self.rhs_kernel = cp.RawKernel(_SOURCE, "boundary_rhs", options=("--std=c++11",))
            compile_cupy_raw(self.moment_kernel)
            compile_cupy_raw(self.rhs_kernel)
            cp.cuda.get_current_stream().synchronize()
        self.device_storage_bytes = sum(a.nbytes for a in self.geometry.values()) + sum(
            a.nbytes for a in (self.partial, self.density, self.rhs))
        self.boundary_tail = None
        self._tail_rows = self._tail_values = None

    def set_boundary_tail(self, tail):
        """Add the rows of an atomic tail of this order to every later RHS."""
        cp, _ = require_cupy()
        if tail is not None and tail.order != self.order:
            raise ValueError(
                f"the atomic tail was built for multipole order {tail.order}, not {self.order}")
        if self._tail_rows is not None:
            self.device_storage_bytes -= self._tail_rows.nbytes + self._tail_values.nbytes
        self.boundary_tail = tail
        self._tail_rows = self._tail_values = None
        if tail is not None:
            with cp.cuda.Device(self.device_id):
                # The rows are distinct, so the indexed add below has no race.
                self._tail_rows = cp.asarray(tail.rows, dtype=(
                    cp.int32 if self.grid.size <= np.iinfo(np.int32).max else cp.int64))
                self._tail_values = cp.asarray(tail.values, dtype=cp.float64)
            self.device_storage_bytes += self._tail_rows.nbytes + self._tail_values.nbytes

    def boundary_check(self, density, count, minimum_order=None, seed=0, tail_charges=None,
                       positions=None):
        """Compare the boundary values of ``density`` with its direct Coulomb sum.

        At ``count`` of the unique exterior stencil points, the sample of
        :func:`_check_points`: the expansion of the moments this builder
        forms, plus the atomic tail of the charges ``tail_charges =
        (positions, charges)`` if the boundary has one, against
        ``2 sum_s rho_s dV/|P-r_s|`` over the grid, summed on the device.
        The atom ``positions`` let the sample take the points nearest to an
        atom.  Returns the points, the number ``total`` they were taken
        from and, in Ry, ``boundary``, ``direct``, ``tail`` and ``legacy``,
        the expansion cut at ``minimum_order``.
        """
        cp, _ = require_cupy()
        from .cupy_atomic_tail import (
            direct_sum_on_device, nearest_atom_on_device, tail_values_on_device)
        _, expansion = self.build_device(density)
        names = ('radius','cosine','sine','phase_real','phase_imag')
        if positions is None and tail_charges is not None:
            positions = tail_charges[0]
        with cp.cuda.Device(self.device_id):
            g = self.geometry
            # These kernels evaluate a point once per stencil entry; the
            # check takes each point once, as the wedge builder holds them.
            unique, _ = unique_rows(np.stack(
                [cp.asnumpy(g['boundary_'+name]) for name in names], axis=1))
            points = _cartesian(*(
                cp.asarray(np.ascontiguousarray(unique[:, column])) for column in range(5)))
            tail = None
            if tail_charges is not None:
                tail = tail_values_on_device(points, *tail_charges, self.order)
            index = cp.asarray(_check_points(
                count, unique[:, 0], None if tail is None else cp.asnumpy(tail),
                None if positions is None else cp.asnumpy(
                    nearest_atom_on_device(points, positions)), seed))
            points = points[index]
            direct = cp.asnumpy(direct_sum_on_device(
                points, self.density, [g['source_'+name] for name in names],
                scale=self.grid.volume_element))
            tail = np.zeros(direct.size) if tail is None else cp.asnumpy(tail[index])
            points = cp.asnumpy(points)
        return dict(
            points=points, total=int(unique.shape[0]), direct=direct, tail=tail,
            boundary=expansion.potential(points)+tail,
            legacy=_truncated_potential(
                expansion, self.order if minimum_order is None else minimum_order, points))

    def build(self, density):
        cp, _ = require_cupy()
        rhs, boundary = self.build_device(density)
        # The borrowed RHS belongs to this builder's device, which need not
        # be the caller's current one.
        with cp.cuda.Device(self.device_id):
            return cp.asnumpy(rhs), boundary

    def build_device(self, density):
        """Borrowed device RHS, valid until the next build on this builder."""
        cp, _ = require_cupy()
        on_device = isinstance(density, cp.ndarray)
        # A device density is converted and checked on this builder's device
        # too: the caller's current one may be another.
        with cp.cuda.Device(self.device_id):
            density = cp.asarray(density,dtype=cp.float64) if on_device else np.ascontiguousarray(density, dtype=np.float64)
            finite = bool(cp.isfinite(density).all().get()) if on_device else np.isfinite(density).all()
            if density.shape != (self.grid.size,) or not finite:
                raise ValueError("density must be finite and match the active grid")
            g = self.geometry
            if on_device:
                cp.copyto(self.density, density)
            else:
                self.density.set(density)
            self.moment_kernel((self.blocks, self.order+1), (256,), (
                np.int64(self.grid.size), np.int32(self.order), np.float64(self.grid.volume_element),
                self.density, *(g['source_'+name] for name in
                ('radius','cosine','sine','phase_real','phase_imag')), g['normalization'], self.partial,
                np.int32(self.stride)))
            positive = self.partial.sum(axis=1)
            self.rhs_kernel((self.blocks,), (256,), (
                np.int64(self.grid.size), np.int32(self.order), self.density, g['boundary_indptr'],
                g['boundary_operator_coefficient'], *(g['boundary_'+name] for name in
                ('radius','cosine','sine','phase_real','phase_imag')), g['normalization'], positive, self.rhs,
                np.int32(self.stride)))
            if self._tail_rows is not None:
                self.rhs[self._tail_rows] += self._tail_values
            host_positive = cp.asnumpy(positive).view(np.complex128).reshape(self.stride,self.stride)
        moments = {}
        for l in range(self.order+1):
            for m in range(l+1):
                value = complex(host_positive[l,m])
                moments[l,m] = value
                if m: moments[l,-m] = (-1)**m*np.conjugate(value)
        return self.rhs, FastMultipoleExpansion(order=self.order, moments=moments)


_SYMMETRY_SOURCE = r'''
#define MAXL 60
__device__ void phase_step(double& re, double& im, double ur, double ui) {
    double next_re = re*ur-im*ui;
    im = re*ui+im*ur;
    re = next_re;
}
// One block reduces span*256 consecutive wedge rows; thread t takes the rows
// base+t+256*s. cls[((l&1)*2+(m&1))*2+{0,1}] weighs Re and Im of the harmonic
// summed over the reflection group.
extern "C" __global__ void moments_sym(
    long long n, int order, int span, const int* m_list, const double* cls,
    const double* rho, const double* weight, const double* radius,
    const double* cosine, const double* sine, const double* phase_re,
    const double* phase_im, const double* norm, double* partial) {
    int m=m_list[blockIdx.y];
    int stride=order+1;
    double acc_re[MAXL+1], acc_im[MAXL+1];
    for(int l=m;l<=order;++l) {acc_re[l]=0.0; acc_im[l]=0.0;}
    long long base=(long long)blockIdx.x*blockDim.x*span+threadIdx.x;
    for(int s=0;s<span;++s) {
        long long row=base+(long long)s*blockDim.x;
        if(row>=n) break;
        double r=radius[row], c=cosine[row], sn=sine[row];
        double ur=phase_re[row], ui=phase_im[row], w=rho[row]*weight[row];
        double pr=1.0, pi=0.0, diagonal=1.0, radial=1.0;
        for(int k=1;k<=m;++k) {
            diagonal*=-(2*k-1)*sn;
            phase_step(pr,pi,ur,ui);
            radial*=r;
        }
        double previous=0.0, current=diagonal;
        for(int l=m;l<=order;++l) {
            if(l==m+1) {previous=diagonal; current=(2*m+1)*c*diagonal;}
            else if(l>m+1) {
                double next=((2*l-1)*c*current-(l+m-1)*previous)/(l-m);
                previous=current; current=next;
            }
            double h=norm[l*stride+m]*current;
            acc_re[l]+=w*radial*(h*pr);
            acc_im[l]-=w*radial*(h*pi);
            radial*=r;
        }
    }
    __shared__ double real_warp[8], imag_warp[8];
    for(int l=m;l<=order;++l) {
        const double* k=cls+((l&1)*2+(m&1))*2;
        double a=k[0]*acc_re[l], b=k[1]*acc_im[l];
        for(int offset=16;offset;offset>>=1) {
            a+=__shfl_down_sync(0xffffffff,a,offset);
            b+=__shfl_down_sync(0xffffffff,b,offset);
        }
        if((threadIdx.x&31)==0) {real_warp[threadIdx.x>>5]=a; imag_warp[threadIdx.x>>5]=b;}
        __syncthreads();
        if(threadIdx.x==0) {
            a=0.0; b=0.0;
            for(int j=0;j<8;++j) {a+=real_warp[j]; b+=imag_warp[j];}
            partial[((l*stride+m)*2)*gridDim.x+blockIdx.x]=a;
            partial[((l*stride+m)*2+1)*gridDim.x+blockIdx.x]=b;
        }
        __syncthreads();
    }
}
// The arithmetic of exterior() in the full-grid kernels, over the m the
// group allows.
__device__ double exterior_sym(double r,double c,double s,double ur,double ui,int order,
                               const int* m_list,int m_count,const double* cls,
                               const double* norm,const double* q) {
    const double pi_const=3.141592653589793238462643383279502884;
    int stride=order+1;
    double pr=1.0,pi=0.0,diagonal=1.0,inverse_m=1.0/r,result=0.0;
    int done=0;
    for(int j=0;j<m_count;++j) {
        int m=m_list[j];
        for(;done<m;++done) {diagonal*=-(2*done+1)*s; phase_step(pr,pi,ur,ui); inverse_m/=r;}
        double previous=0.0,current=diagonal,radial=inverse_m;
        for(int l=m;l<=order;++l) {
            if(l==m+1) {previous=diagonal; current=(2*m+1)*c*diagonal;}
            else if(l>m+1) {
                double next=((2*l-1)*c*current-(l+m-1)*previous)/(l-m);
                previous=current; current=next;
            }
            const double* k=cls+((l&1)*2+(m&1))*2;
            if(k[0]!=0.0||k[1]!=0.0) {
                double h=norm[l*stride+m]*current;
                int index=(l*stride+m)*2;
                double positive=q[index]*(h*pr)-q[index+1]*(h*pi);
                result+=(4*pi_const/(2*l+1)*radial)*(m?2*positive:positive);
            }
            radial/=r;
        }
    }
    return 2*result;
}
// Boundary value of each unique exterior point; the static atomic tail of
// the point is added where there is one.
extern "C" __global__ void exterior_values_sym(
    long long n,int order,const int* m_list,int m_count,const double* cls,
    const double* radius,const double* cosine,const double* sine,
    const double* phase_re,const double* phase_im,
    const double* norm,const double* q,int has_tail,const double* tail,double* value) {
    long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n) return;
    double v=exterior_sym(radius[p],cosine[p],sine[p],phase_re[p],phase_im[p],
                          order,m_list,m_count,cls,norm,q);
    value[p]=has_tail?v+tail[p]:v;
}
// rhs[w] = sqrt(m_w) (8 pi rho_w - sum_t a_t V[p_t]): the normalized wedge RHS.
extern "C" __global__ void assemble_rhs_sym(
    long long n,const double* rho,const double* root,const long long* indptr,
    const double* coefficient,const int* point,const double* value,double* rhs) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    double v=8*3.141592653589793238462643383279502884*rho[row];
    for(long long t=indptr[row];t<indptr[row+1];++t) v-=coefficient[t]*value[point[t]];
    rhs[row]=v*root[row];
}
'''
_MOMENT_SPAN = 16


def reflection_class_table(signs):
    """Weights of ``Re Y_lm`` and ``Im Y_lm`` summed over a reflection group.

    For the group of the diagonal operations ``signs`` (rows ``(sx,sy,sz)``)
    and a function invariant under it, the moments are sums over one point
    ``w`` per orbit with multiplicity ``m_w``:

    ``Q_lm = sum_w f_w m_w r_w**l N_lm P_l^m [cR cos(m phi) - i cI sin(m phi)]``.

    ``table[l%2, m%2] = (cR, cI)``: with ``eps_g = (-1)**m`` if ``sx<0``
    times ``(-1)**(l+m)`` if ``sz<0``, ``a`` the sum of ``eps_g`` over the
    operations with ``sx*sy>0`` and ``b`` over the others,
    ``cR = (a+b)/|G|`` and ``cI = (a-b)/|G|``.
    """
    signs = np.asarray(signs, dtype=np.int64).reshape(-1, 3)
    table = np.zeros((2, 2, 2), dtype=np.float64)
    for l_parity in (0, 1):
        for m_parity in (0, 1):
            a = b = 0
            for sx, sy, sz in signs:
                eps = (1 if sx > 0 else (-1)**m_parity)*(1 if sz > 0 else (-1)**(l_parity+m_parity))
                if sx*sy > 0: a += eps
                else: b += eps
            table[l_parity, m_parity] = ((a+b)/len(signs), (a-b)/len(signs))
    return table


def wedge_kernels_supported(reduction):
    """Whether ``reduction`` is a group of axis reflections with its signs."""
    return reduction is not None and not isinstance(reduction, SignedPermutationReduction)


class CuPySymmetryMultipoleBoundaryBuilder:
    """Isolated boundary on the wedge of an axis-reflection group.

    The moments are summed over one row per orbit with the parity classes of
    :func:`reflection_class_table`, the boundary values are evaluated once
    per unique exterior point of the stencils of those rows, and the
    normalized wedge right-hand side ``U.T b`` is gathered from them: the
    contract of ``NativeSymmetryMultipoleBoundaryBuilder.build_reduced``.
    Moments the group forbids are exact zeros.  The results differ from the
    full-grid kernels at round-off.
    """

    def __init__(self, grid, reduction, multipole_order=9, device_id=None):
        cp, _ = require_cupy()
        if not wedge_kernels_supported(reduction):
            raise ValueError("the wedge boundary kernels need an axis-reflection reduction")
        if reduction.full_size != grid.size:
            raise ValueError("symmetry reduction does not match the Hartree grid")
        native = NativeMultipoleBoundaryBuilder(grid, multipole_order)
        if not hasattr(native._native_builder, "export_full_geometry"):
            raise RuntimeError("GPU boundary requires the rebuilt native geometry exporter")
        self.grid, self.reduction, self.order = grid, reduction, native.order
        self.stride = self.order+1
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        payload = native._native_builder.export_full_geometry()
        del native
        names = ('radius', 'cosine', 'sine', 'phase_real', 'phase_imag')
        rows = np.asarray(reduction.representative_rows, dtype=np.int64)
        wedge = int(rows.size)
        # Stencil entries of the representative rows, in wedge order.
        indptr = np.asarray(payload['boundary_indptr'], dtype=np.int64)
        counts = indptr[rows+1]-indptr[rows]
        wedge_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        terms = np.repeat(indptr[rows]-wedge_indptr[:-1], counts)+np.arange(wedge_indptr[-1])
        self.boundary_term_count = int(terms.size)
        # A point reached from several rows has one bit pattern of geometry.
        unique, point = unique_rows(
            np.stack([np.asarray(payload['boundary_'+name])[terms] for name in names], axis=1))
        self.exterior_point_count = int(unique.shape[0])
        cls = reflection_class_table(reduction.signs)
        m_list = np.array([m for m in range(self.stride) if cls[:, m & 1].any()], dtype=np.int32)
        multiplicities = np.asarray(reduction.multiplicities, dtype=np.float64)
        self.blocks = (wedge+256*_MOMENT_SPAN-1)//(256*_MOMENT_SPAN)
        with cp.cuda.Device(self.device_id):
            self.source = [cp.asarray(np.asarray(payload['source_'+name])[rows]) for name in names]
            self.weight = cp.asarray(multiplicities*grid.volume_element)
            self.root = cp.asarray(np.sqrt(multiplicities))
            self.indptr = cp.asarray(wedge_indptr)
            self.coefficient = cp.asarray(np.asarray(payload['boundary_operator_coefficient'])[terms])
            self.point = cp.asarray(point.reshape(-1), dtype=cp.int32)
            self.exterior = [cp.asarray(np.ascontiguousarray(unique[:, column])) for column in range(5)]
            self.norm = cp.asarray(_stride_normalization(payload['normalization'], self.order))
            self.cls = cp.asarray(cls.reshape(-1))
            self.m_list = cp.asarray(m_list)
            del payload, unique, terms
            self.partial = cp.zeros((2*self.stride*self.stride, self.blocks), dtype=cp.float64)
            self.density = cp.empty(wedge, dtype=cp.float64)
            self.rhs = cp.empty(wedge, dtype=cp.float64)
            self.value = cp.empty(self.exterior_point_count, dtype=cp.float64)
            self.tail = None
            self.kernels = tuple(
                cp.RawKernel(_SYMMETRY_SOURCE, name, options=("--std=c++11",))
                for name in ("moments_sym", "exterior_values_sym", "assemble_rhs_sym"))
            for kernel in self.kernels:
                compile_cupy_raw(kernel)
            cp.cuda.get_current_stream().synchronize()
        self.atomic_tail_maximum = None
        self.atomic_tail_seconds = 0.0
        self.atomic_tail_values = None
        self.device_storage_bytes = sum(a.nbytes for a in (
            *self.source, *self.exterior, self.weight, self.root, self.indptr, self.coefficient,
            self.point, self.norm, self.cls, self.m_list, self.partial, self.density, self.rhs,
            self.value))

    def exterior_points_device(self):
        """Cartesian coordinates of the unique exterior points, on the device."""
        return _cartesian(*self.exterior)

    def set_atomic_tail(self, positions, charges, tail_values=None):
        """Fold the static atomic tail of these point charges into the values.

        ``C_L`` is evaluated once per unique exterior point on the device and
        added to the boundary value of the point in every later build;
        ``positions=None`` removes it.  ``tail_values(points, positions,
        charges, order)`` evaluates it at host points instead, as in
        :func:`.atomic_tail.build_atomic_tail_fast`.  ``atomic_tail_values``
        says which of the two gave the values the builder holds.
        """
        cp, _ = require_cupy()
        started = perf_counter()
        if self.tail is not None:
            self.device_storage_bytes -= self.tail.nbytes
        self.tail, self.atomic_tail_maximum, self.atomic_tail_seconds = None, None, 0.0
        self.atomic_tail_values = None
        if positions is None:
            return
        from .cupy_atomic_tail import tail_values_on_device
        with cp.cuda.Device(self.device_id):
            points = self.exterior_points_device()
            if tail_values is None:
                self.tail = tail_values_on_device(points, positions, charges, self.order)
            else:
                self.tail = cp.asarray(np.ascontiguousarray(tail_values(
                    cp.asnumpy(points), positions, charges, self.order), dtype=np.float64))
            self.atomic_tail_maximum = float(cp.abs(self.tail).max().get())
        self.atomic_tail_values = "device kernel" if tail_values is None else "host threads"
        self.device_storage_bytes += self.tail.nbytes
        self.atomic_tail_seconds = perf_counter()-started

    def boundary_check(self, wedge_density, count, minimum_order=None, seed=0, positions=None):
        """Compare the boundary values of a density with its direct Coulomb sum.

        At ``count`` of the unique exterior points, the sample of
        :func:`_check_points`: the values the kernels gather the right-hand
        side from, against ``2 sum_s rho_s dV/|P-r_s|`` over the full grid,
        summed on the device from the wedge rows and their images under the
        group.  The atom ``positions`` let the sample take the points
        nearest to an atom.  Returns the points, the number ``total`` they
        were taken from and, in Ry, ``boundary``, ``direct``, ``tail`` and
        ``legacy``, the expansion cut at ``minimum_order``.
        """
        cp, _ = require_cupy()
        from .cupy_atomic_tail import direct_sum_on_device, nearest_atom_on_device
        _, expansion = self.build_reduced_device(wedge_density)
        with cp.cuda.Device(self.device_id):
            points = self.exterior_points_device()
            index = cp.asarray(_check_points(
                count, cp.asnumpy(self.exterior[0]),
                None if self.tail is None else cp.asnumpy(self.tail),
                None if positions is None else cp.asnumpy(
                    nearest_atom_on_device(points, positions)), seed))
            points = points[index]
            # A wedge row stands for m_w grid points: each of its |G| images
            # carries m_w/|G| of the weight of one.
            direct = cp.asnumpy(direct_sum_on_device(
                points, self.density, self.source, weight=self.weight,
                scale=1.0/self.reduction.group_order, signs=self.reduction.signs))
            boundary = cp.asnumpy(self.value[index])
            tail = np.zeros(direct.size) if self.tail is None else cp.asnumpy(self.tail[index])
            points = cp.asnumpy(points)
        return dict(
            points=points, total=self.exterior_point_count, direct=direct, tail=tail,
            boundary=boundary,
            legacy=_truncated_potential(
                expansion, self.order if minimum_order is None else minimum_order, points))

    def build_reduced(self, density):
        """Host interface: normalized wedge RHS and the multipole expansion."""
        cp, _ = require_cupy()
        if isinstance(density, SymmetryScalarField):
            if density.reduction is self.reduction:
                wedge_density = density.values
            else:
                wedge_density = self.reduction.reduce_vector(np.ascontiguousarray(
                    density.values[density.reduction.full_to_wedge]))/np.sqrt(self.reduction.multiplicities)
        else:
            density = np.asarray(density, dtype=np.float64)
            if density.shape != (self.grid.size,):
                raise ValueError("density does not match the active grid")
            # The orbit average, as the native wedge builder takes it.
            wedge_density = self.reduction.reduce_vector(density)/np.sqrt(self.reduction.multiplicities)
        rhs, boundary = self.build_reduced_device(wedge_density)
        with cp.cuda.Device(self.device_id):
            return cp.asnumpy(rhs), boundary

    def build_reduced_device(self, wedge_density):
        """Borrowed device RHS of one physical value per orbit, valid until the next build."""
        cp, _ = require_cupy()
        on_device = isinstance(wedge_density, cp.ndarray)
        wedge = self.reduction.wedge_size
        with cp.cuda.Device(self.device_id):
            density = (cp.asarray(wedge_density, dtype=cp.float64) if on_device
                       else np.ascontiguousarray(wedge_density, dtype=np.float64))
            finite = bool(cp.isfinite(density).all().get()) if on_device else np.isfinite(density).all()
            if density.shape != (wedge,) or not finite:
                raise ValueError("density must be finite and match the symmetry wedge")
            if on_device:
                cp.copyto(self.density, density)
            else:
                self.density.set(density)
            moments, values, assemble = self.kernels
            count = np.int32(self.m_list.size)
            moments((self.blocks, int(self.m_list.size)), (256,), (
                np.int64(wedge), np.int32(self.order), np.int32(_MOMENT_SPAN), self.m_list, self.cls,
                self.density, self.weight, *self.source, self.norm, self.partial))
            positive = self.partial.sum(axis=1)
            points = self.exterior_point_count
            values(((points+255)//256,), (256,), (
                np.int64(points), np.int32(self.order), self.m_list, count, self.cls,
                *self.exterior, self.norm, positive, np.int32(self.tail is not None),
                self.value if self.tail is None else self.tail, self.value))
            assemble(((wedge+255)//256,), (256,), (
                np.int64(wedge), self.density, self.root, self.indptr, self.coefficient,
                self.point, self.value, self.rhs))
            host_positive = cp.asnumpy(positive).view(np.complex128).reshape(self.stride, self.stride)
        moments = {}
        for l in range(self.order+1):
            for m in range(l+1):
                value = complex(host_positive[l, m])
                moments[l, m] = value
                if m: moments[l, -m] = (-1)**m*np.conjugate(value)
        return self.rhs, FastMultipoleExpansion(order=self.order, moments=moments)


def identity_reduction(size):
    """The reduction of the group of the identity alone: every row is its own orbit."""
    rows = np.arange(size, dtype=np.int64)
    return AxisReflectionReduction(
        signs=np.ones((1, 3), dtype=np.int8), representative_rows=rows,
        full_to_wedge=rows, multiplicities=np.ones(size, dtype=np.int64))


class CuPyPointMultipoleBoundaryBuilder:
    """Full-grid isolated boundary with one value per unique exterior point.

    The kernels of :class:`CuPySymmetryMultipoleBoundaryBuilder` under the
    identity alone: the moments are summed over every grid row, without a
    moment the group could forbid, and the boundary values are evaluated
    once per unique exterior point, which four to five stencil entries
    share.  It has the full-grid interface of
    :class:`CuPyMultipoleBoundaryBuilder`, whose results it reproduces to
    round-off, and takes the atomic tail per point like the wedge builder.
    """

    def __init__(self, grid, multipole_order=9, device_id=None):
        self._wedge = CuPySymmetryMultipoleBoundaryBuilder(
            grid, identity_reduction(grid.size), multipole_order, device_id=device_id)
        self.grid, self.order = grid, self._wedge.order
        self.device_id = self._wedge.device_id
        self.boundary_term_count = self._wedge.boundary_term_count
        self.exterior_point_count = self._wedge.exterior_point_count

    @property
    def device_storage_bytes(self):
        return self._wedge.device_storage_bytes

    @property
    def atomic_tail_maximum(self):
        return self._wedge.atomic_tail_maximum

    @property
    def atomic_tail_seconds(self):
        return self._wedge.atomic_tail_seconds

    @property
    def atomic_tail_values(self):
        return self._wedge.atomic_tail_values

    def set_atomic_tail(self, positions, charges, tail_values=None):
        """Fold the static atomic tail of these point charges into the values."""
        self._wedge.set_atomic_tail(positions, charges, tail_values)

    def boundary_check(self, density, count, minimum_order=None, seed=0, positions=None):
        """The check of the wedge builder, whose wedge is the grid here."""
        return self._wedge.boundary_check(
            density, count, minimum_order=minimum_order, seed=seed, positions=positions)

    def build(self, density):
        cp, _ = require_cupy()
        rhs, boundary = self.build_device(density)
        with cp.cuda.Device(self.device_id):
            return cp.asnumpy(rhs), boundary

    def build_device(self, density):
        """Borrowed device RHS, valid until the next build on this builder."""
        return self._wedge.build_reduced_device(density)
