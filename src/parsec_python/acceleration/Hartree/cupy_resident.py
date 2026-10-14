"""Experimental device-resident boundary/projection/chronological CG chain.

The SCF scalar interface remains on the host wedge. Only wedge density enters
and wedge potential/RHS leave this chain; full-grid fields stay on the GPU.
"""
from time import perf_counter
import os
import numpy as np
from ..backends.cupy import require_cupy
from ..SCF.symmetry_fields import SymmetryScalarField
from .native_poisson import NativePoissonResult

_SOURCE=r'''
extern "C" __global__ void reduce_ordered(int n,const int* ptr,const int* members,
 const double* roots,const double* full,double* out) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
 double sum=0;
 for(int k=ptr[row];k<ptr[row+1];++k)sum+=full[members[k]];
 out[row]=sum/roots[row];
}
extern "C" __global__ void physical_average(int n,const int* ptr,const double* roots,
 const double* normalized,const double* rhs,double* out) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
 int multiplicity=ptr[row+1]-ptr[row];
 double v=normalized[row]/roots[row],b=rhs[row]/roots[row];
 double sv=0,sb=0;
 // Reproduce expand_vector -> np.bincount -> divide, without full arrays.
 for(int k=0;k<multiplicity;++k){sv+=v;sb+=b;}
 out[row]=sv/multiplicity;out[n+row]=sb/multiplicity;
}
'''


class CuPyResidentHartree:
    def __init__(self,builder,reduction,backend,settings):
        cp,_=require_cupy();self.cp=cp;self.builder=builder;self.reduction=reduction
        self.backend=backend;self.settings=settings;self.history=[]
        self.predictor=os.environ.get('PARSEC_CUPY_RESIDENT_PREDICTOR','host')
        if self.predictor not in ('host','device'):raise ValueError('resident predictor must be host or device')
        if reduction.full_size>np.iinfo(np.int32).max:raise ValueError('resident projection needs int32 grid')
        # A wedge builder returns the normalized wedge RHS itself: the two
        # full-grid maps are then uploaded only if a full-grid array arrives.
        self.wedge_builder=hasattr(builder,'build_reduced_device')
        self.map=self.members=None
        with cp.cuda.Device(builder.device_id):
            self.roots=cp.asarray(np.sqrt(reduction.multiplicities))
            self.ptr=cp.asarray(np.r_[0,np.cumsum(reduction.multiplicities)],dtype=cp.int32)
            if not self.wedge_builder:self._upload_full_maps()
            self.kernel=cp.RawKernel(_SOURCE,'reduce_ordered',options=('--std=c++11','--fmad=false'))
            self.kernel.compile()
            self.export_kernel=cp.RawKernel(_SOURCE,'physical_average',options=('--std=c++11','--fmad=false'))
            self.export_kernel.compile()
        self.rhs_seconds=self.solve_seconds=0.

    def _upload_full_maps(self):
        cp=self.cp;r=self.reduction
        self.map=cp.asarray(r.full_to_wedge,dtype=cp.int32)
        self.members=cp.asarray(np.argsort(r.full_to_wedge,kind='stable'),dtype=cp.int32)

    def project(self,full):
        if self.members is None:self._upload_full_maps()
        out=self.cp.empty(self.reduction.wedge_size,self.cp.float64)
        self.kernel(((out.size+255)//256,),(256,),
                    (np.int32(out.size),self.ptr,self.members,self.roots,full,out))
        return out

    def solve(self,density,initial_potential=None,**kwargs):
        cp=self.cp;r=self.reduction
        with cp.cuda.Device(self.builder.device_id):
            start=perf_counter()
            if isinstance(density,SymmetryScalarField):
                if density.reduction is not r:raise ValueError('resident density uses another wedge')
                device_density=cp.asarray(density.values)
                if not self.wedge_builder:device_density=device_density[self.map]
            else:
                device_density=cp.asarray(density)
                # One physical value per orbit: the orbit average.
                if self.wedge_builder:device_density=self.project(device_density)/self.roots
            if self.wedge_builder:
                # The builder lends its buffer; the predictor keeps two RHS.
                rhs,boundary=self.builder.build_reduced_device(device_density)
                rhs=rhs.copy()
            else:
                full_rhs,boundary=self.builder.build_device(device_density)
                rhs=self.project(full_rhs)
            cp.cuda.get_current_stream().synchronize();self.rhs_seconds=perf_counter()-start
            start=perf_counter()
            host_rhs=cp.asnumpy(rhs) if self.predictor=='host' else None
            if initial_potential is None: initial=cp.zeros_like(rhs)
            elif isinstance(initial_potential,SymmetryScalarField):
                if initial_potential.reduction is not r:raise ValueError('resident initial uses another wedge')
                initial=cp.asarray(initial_potential.values)*self.roots
            else: initial=self.project(cp.asarray(initial_potential))
            if len(self.history)==2 and os.environ.get('PARSEC_HARTREE_CHRONOLOGICAL_GUESS','1').lower() not in {'0','false','no','off'}:
                old_rhs,old_x=self.history[0];prev_rhs,prev_x=self.history[1]
                xp=np if self.predictor=='host' else cp
                direction=prev_rhs-old_rhs;denominator=xp.dot(direction,direction)
                host_den=float(denominator) if self.predictor=='host' else float(denominator.get())
                if np.isfinite(host_den) and host_den>0:
                    new_rhs=host_rhs if self.predictor=='host' else rhs
                    alpha=xp.clip(xp.dot(direction,new_rhs-prev_rhs)/denominator,-.5,1.5)
                    predicted=prev_x+alpha*(prev_x-old_x)
                    # Match the original chronological predictor's guard.
                    if bool(xp.all(xp.isfinite(predicted))):
                        initial=predicted
            controls={name:kwargs.pop(name,getattr(self.settings,name)) for name in
                      ('relative_tolerance','absolute_tolerance','max_iterations')}
            raise_on_nonconvergence=kwargs.pop('raise_on_nonconvergence',True)
            if kwargs:raise TypeError(f'Unsupported resident Hartree controls: {tuple(kwargs)}')
            payload=self.backend.solve(rhs,initial,device_result=True,**controls)
            if not payload['converged'] and raise_on_nonconvergence:raise RuntimeError('resident Hartree CG did not converge')
            solution=payload.pop('solution')
            self.history.append((host_rhs,cp.asnumpy(solution)) if self.predictor=='host' else (rhs,solution))
            self.history=self.history[-2:]
            # Existing CPU scalar algebra receives physical point values.
            if self.predictor=='host':
                physical=cp.empty((2,r.wedge_size),cp.float64)
                self.export_kernel(((r.wedge_size+255)//256,),(256,),
                    (np.int32(r.wedge_size),self.ptr,self.roots,solution,rhs,physical))
                host=cp.asnumpy(physical)
            else:
                host=cp.asnumpy(cp.stack((solution/self.roots,rhs/self.roots)))
            self.solve_seconds=perf_counter()-start
            result=NativePoissonResult(potential=SymmetryScalarField(r,host[0]),
                right_hand_side=SymmetryScalarField(r,host[1]),**payload)
            return result.as_hartree_result(boundary)
