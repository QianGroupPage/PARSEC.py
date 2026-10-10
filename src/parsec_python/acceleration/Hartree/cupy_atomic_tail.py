"""FP64 device kernel for the static atomic tail of the Hartree boundary.

One thread per exterior point sums ``2 q_a/|P-R_a|`` over the atoms in input
order with a compensated sum, and subtracts the order-``L`` series of the
same charges, evaluated with the recurrences of the boundary kernels.  The
kernel is compiled without fused multiply-adds, which would cancel the
compensation.  It is launched once per prepared system.
"""
from __future__ import annotations

import numpy as np

from parsec_python.Hartree.harmonics import _normalization

from ..backends.cupy import require_cupy
from ..backends.cupy_compile import compile_cupy_raw
from .atomic_tail import point_charge_moments


_SOURCE = r'''
__device__ void phase_step(double& re, double& im, double ur, double ui) {
    double next_re = re*ur-im*ui;
    im = re*ui+im*ur;
    re = next_re;
}
__device__ double series(double x,double y,double z,int order,
                         const double* norm,const double* q) {
    const double pi_const=3.141592653589793238462643383279502884;
    int stride=order+1;
    double r=sqrt(x*x+y*y+z*z);
    double c=fmin(fmax(z/r,-1.0),1.0);
    double s=sqrt(fmax(0.0,1.0-c*c));
    double xy=hypot(x,y);
    double ur=1.0,ui=0.0;
    if(xy>0.0) {ur=x/xy; ui=y/xy;}
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
// order < 0 leaves the Coulomb sum alone.
extern "C" __global__ void atomic_tail(
    long long n,const double* points,int atom_count,const double* atoms,
    const double* charges,int order,const double* norm,const double* q,
    double* out) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    double x=points[3*row],y=points[3*row+1],z=points[3*row+2];
    double total=0.0,carry=0.0;
    for(int a=0;a<atom_count;++a) {
        double dx=x-atoms[3*a],dy=y-atoms[3*a+1],dz=z-atoms[3*a+2];
        double term=charges[a]/sqrt(dx*dx+dy*dy+dz*dz)-carry;
        double next=total+term;
        carry=(next-total)-term;
        total=next;
    }
    double value=2.0*total;
    if(order>=0) value-=series(x,y,z,order,norm,q);
    out[row]=value;
}
// Distance from each point to the atom nearest to it.
extern "C" __global__ void nearest_atom(
    long long n,const double* points,int atom_count,const double* atoms,
    double* out) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    double x=points[3*row],y=points[3*row+1],z=points[3*row+2];
    double nearest=1.0e300;
    for(int a=0;a<atom_count;++a) {
        double dx=x-atoms[3*a],dy=y-atoms[3*a+1],dz=z-atoms[3*a+2];
        nearest=fmin(nearest,dx*dx+dy*dy+dz*dz);
    }
    out[row]=sqrt(nearest);
}
// Coulomb sum of grid sources begin..end at the points, continued over
// several launches through total and carry. A source is given by the
// angular geometry of the boundary builders; each is counted at its images
// under the sign triples.
extern "C" __global__ void direct_sum(
    long long n,const double* points,long long begin,long long end,
    const double* rho,int has_weight,const double* weight,double scale,
    const double* radius,const double* cosine,const double* sine,
    const double* phase_re,const double* phase_im,
    int images,const double* signs,double* total,double* carry) {
    long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n) return;
    double x=points[3*p],y=points[3*p+1],z=points[3*p+2];
    double sum=total[p],lost=carry[p];
    for(long long s=begin;s<end;++s) {
        double w=rho[s]*scale;
        if(has_weight) w*=weight[s];
        double planar=radius[s]*sine[s];
        double sx=planar*phase_re[s],sy=planar*phase_im[s],sz=radius[s]*cosine[s];
        for(int g=0;g<images;++g) {
            double dx=x-signs[3*g]*sx,dy=y-signs[3*g+1]*sy,dz=z-signs[3*g+2]*sz;
            double term=w/sqrt(dx*dx+dy*dy+dz*dz)-lost;
            double next=sum+term;
            lost=(next-sum)-term;
            sum=next;
        }
    }
    total[p]=sum; carry[p]=lost;
}
'''
_KERNEL = None
_DIRECT_KERNEL = None
_NEAREST_KERNEL = None
# Sources per launch of the direct sum, which bounds the time of one launch.
_DIRECT_SOURCES_PER_LAUNCH = 1 << 21


def _kernel(cp):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(
            _SOURCE, "atomic_tail", options=("--std=c++11", "--fmad=false")
        )
    compile_cupy_raw(_KERNEL)
    return _KERNEL


def normalization_table(order: int) -> np.ndarray:
    """Return the ``Y_lm`` prefactors at ``[l*(order+1)+m]``, zero for ``m>l``."""

    stride = order + 1
    table = np.zeros(stride * stride, dtype=np.float64)
    for angular_momentum in range(stride):
        for magnetic in range(angular_momentum + 1):
            table[angular_momentum * stride + magnetic] = _normalization(
                angular_momentum, magnetic
            )
    return table


def tail_values_on_device(points, positions, charges, order: int):
    """Return ``C_L`` at device points of shape ``(n, 3)`` as a device array.

    ``order < 0`` returns the Coulomb sum ``2 sum_a q_a/|P-R_a|`` alone.
    The caller has made the device of ``points`` current.
    """

    cp, _ = require_cupy()
    positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.ascontiguousarray(charges, dtype=np.float64)
    points = cp.ascontiguousarray(points, dtype=cp.float64)
    count = int(points.shape[0])
    out = cp.empty(count, dtype=cp.float64)
    if order >= 0:
        norm = cp.asarray(normalization_table(order))
        moments = cp.asarray(
            np.ascontiguousarray(
                point_charge_moments(positions, charges, order)
            ).view(np.float64).reshape(-1)
        )
    else:
        norm = moments = cp.zeros(1, dtype=cp.float64)
    if count:
        _kernel(cp)(
            ((count + 127) // 128,),
            (128,),
            (
                np.int64(count), points, np.int32(positions.shape[0]),
                cp.asarray(positions), cp.asarray(charges), np.int32(order),
                norm, moments, out,
            ),
        )
    return out


def nearest_atom_on_device(points, positions):
    """Return the distance from each device point to its nearest atom (bohr).

    The caller has made the device of ``points`` current.
    """

    global _NEAREST_KERNEL
    cp, _ = require_cupy()
    if _NEAREST_KERNEL is None:
        _NEAREST_KERNEL = cp.RawKernel(
            _SOURCE, "nearest_atom", options=("--std=c++11", "--fmad=false")
        )
    compile_cupy_raw(_NEAREST_KERNEL)
    positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
    points = cp.ascontiguousarray(points, dtype=cp.float64)
    count = int(points.shape[0])
    out = cp.empty(count, dtype=cp.float64)
    if count:
        _NEAREST_KERNEL(
            ((count + 127) // 128,),
            (128,),
            (
                np.int64(count), points, np.int32(positions.shape[0]),
                cp.asarray(positions), out,
            ),
        )
    return out


def direct_sum_on_device(points, density, geometry, *, weight=None, scale=1.0, signs=None):
    """Return ``2 sum_s w_s sum_g 1/|P - g r_s|`` (Ry) at device points.

    ``density`` and the five ``geometry`` arrays (radius, cosine, sine and
    the two phase components) describe the sources as the boundary builders
    hold them; ``w_s = density_s * scale * weight_s``.  ``signs`` lists the
    sign triples ``g`` whose images of every source are summed, the identity
    alone by default.  The caller has made the device of the arrays current.
    """

    global _DIRECT_KERNEL
    cp, _ = require_cupy()
    if _DIRECT_KERNEL is None:
        _DIRECT_KERNEL = cp.RawKernel(
            _SOURCE, "direct_sum", options=("--std=c++11", "--fmad=false")
        )
    compile_cupy_raw(_DIRECT_KERNEL)
    points = cp.ascontiguousarray(points, dtype=cp.float64)
    count, sources = int(points.shape[0]), int(density.shape[0])
    images = np.ones((1, 3)) if signs is None else np.asarray(signs, dtype=np.float64)
    device_signs = cp.asarray(np.ascontiguousarray(images.reshape(-1, 3)))
    total = cp.zeros(count, dtype=cp.float64)
    carry = cp.zeros(count, dtype=cp.float64)
    for begin in range(0, sources, _DIRECT_SOURCES_PER_LAUNCH):
        _DIRECT_KERNEL(
            ((count + 63) // 64,),
            (64,),
            (
                np.int64(count), points, np.int64(begin),
                np.int64(min(sources, begin + _DIRECT_SOURCES_PER_LAUNCH)),
                density, np.int32(weight is not None),
                density if weight is None else weight, np.float64(scale),
                *geometry, np.int32(device_signs.shape[0]), device_signs,
                total, carry,
            ),
        )
    return 2.0 * total


def device_tail_values(points, positions, charges, order: int, *, device_id=None):
    """Host interface of the kernel: ``C_L`` at host points, on one device."""

    cp, _ = require_cupy()
    device = int(cp.cuda.Device().id if device_id is None else device_id)
    with cp.cuda.Device(device):
        return cp.asnumpy(
            tail_values_on_device(
                cp.asarray(np.ascontiguousarray(points, dtype=np.float64)),
                positions,
                charges,
                order,
            )
        )


def device_coulomb_sum(points, positions, charges, *, device_id=None):
    """Return ``2 sum_a q_a/|P-R_a|`` at host points, summed on one device."""

    return device_tail_values(points, positions, charges, -1, device_id=device_id)


__all__ = [
    "device_coulomb_sum",
    "device_tail_values",
    "direct_sum_on_device",
    "nearest_atom_on_device",
    "normalization_table",
    "tail_values_on_device",
]
