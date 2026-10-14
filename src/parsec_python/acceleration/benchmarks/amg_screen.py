"""AMG-PCG feasibility test on captured unchanged production Poisson systems.

Hierarchy construction is timed and reused across right-hand sides. This is
a screening tool, not a replacement for the production stopping policy.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cs
import pyamg
from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedConjugateGradientBackend

_CSR_SOURCE=r'''
extern "C" __global__ void csr_warp(int n,const int* ptr,const int* col,
 const double* a,const double* x,double* y){
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
 if(row>=n)return;
 double sum=0;
 for(int k=ptr[row]+lane;k<ptr[row+1];k+=32)sum+=a[k]*x[col[k]];
 for(int delta=16;delta>0;delta/=2)sum+=__shfl_down_sync(0xffffffff,sum,delta);
 if(lane==0)y[row]=sum;
}
extern "C" __global__ void dense_inverse(int n,const double* a,const double* b,double* x){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
 double sum=0;for(int k=0;k<n;++k)sum+=a[row*n+k]*b[k];x[row]=sum;
}
'''

class GraphCSR:
    """Capture-compatible preconditioner SpMV; original fine operator is retained."""
    kernel=None
    def __init__(self,matrix):
        self.native=matrix
        self.data,self.indices,self.indptr=matrix.data,matrix.indices,matrix.indptr
        self.shape=matrix.shape
        if self.indices.dtype!=cp.int32 or self.indptr.dtype!=cp.int32:
            raise ValueError('graph screen supports int32 CSR only')
        if GraphCSR.kernel is None:
            GraphCSR.kernel=cp.RawKernel(_CSR_SOURCE,'csr_warp',options=('--std=c++11','--fmad=false'))
            GraphCSR.kernel.compile()
    def __matmul__(self,x):
        y=cp.empty(self.shape[0],cp.float64)
        self.kernel(((self.shape[0]*32+255)//256,),(256,),
                    (np.int32(self.shape[0]),self.indptr,self.indices,self.data,x,y))
        return y


def main():
    p=argparse.ArgumentParser();p.add_argument('capture');p.add_argument('output');p.add_argument('--geometric-capture');p.add_argument('--capture-spmv',action='store_true');a=p.parse_args()
    files=sorted(Path(a.capture).glob('poisson_*.npz'))
    if not files: raise RuntimeError('no captured systems')
    first=np.load(files[0]); host=sp.csr_matrix((first['data'],first['indices'],first['indptr']),shape=tuple(first['shape']))
    t=time.perf_counter()
    if a.geometric_capture:
        from parsec_python.acceleration.benchmarks.geometric_amg import hierarchy_from_capture
        hierarchy=hierarchy_from_capture(host,a.geometric_capture)
    else:
        hierarchy=pyamg.smoothed_aggregation_solver(host,symmetry='hermitian',max_coarse=96,max_levels=8,
                    presmoother=('jacobi',{'omega':4/3,'iterations':2}),
                    postsmoother=('jacobi',{'omega':4/3,'iterations':2}))
    build_s=time.perf_counter()-t
    print('HIERARCHY_LEVELS',[l.A.shape for l in hierarchy.levels],flush=True)
    t=time.perf_counter(); levels=[]
    for lev in hierarchy.levels:
        matrix=cs.csr_matrix(lev.A)
        item={'a':matrix,'di':cp.asarray(1/lev.A.diagonal())}
        if hasattr(lev,'P'):
            item.update(p=cs.csr_matrix(lev.P),r=cs.csr_matrix(lev.R))
            # Use the same diagonally scaled spectral-radius bound for both
            # pre/post Jacobi sweeps, keeping the V-cycle symmetric.
            scaled=sp.diags(1/lev.A.diagonal())@lev.A
            bound=float(np.max(np.asarray(abs(scaled).sum(axis=1)).ravel()))
            item['omega']=.8/bound
        levels.append(item)
    original_fine=levels[0]['a']
    spmv_max_error=0.
    if a.capture_spmv:
        for lev in levels:
            for key in ('a','p','r'):
                if key not in lev:continue
                native=lev[key];wrapped=GraphCSR(native)
                x=cp.asarray(np.random.default_rng(61).normal(size=native.shape[1]))
                error=float(cp.max(cp.abs(wrapped@x-native@x)).get())
                scale=float(cp.max(cp.abs(native@x)).get())
                if error>5e-13*max(1.,scale):raise RuntimeError('custom preconditioner SpMV mismatch')
                spmv_max_error=max(error,spmv_max_error);lev[key]=wrapped
    inverse=cp.asarray(np.linalg.pinv(hierarchy.levels[-1].A.toarray(),hermitian=True))
    if a.capture_spmv:
        inverse=cp.ascontiguousarray(inverse)
        dense_kernel=cp.RawKernel(_CSR_SOURCE,'dense_inverse',options=('--std=c++11','--fmad=false'))
        dense_kernel.compile()
    cp.cuda.get_current_stream().synchronize(); upload_s=time.perf_counter()-t
    def cycle(b,k=0):
        if k==len(levels)-1:
            if not a.capture_spmv:return inverse@b
            result=cp.empty_like(b)
            dense_kernel(((b.size+127)//128,),(128,),(np.int32(b.size),inverse,b,result))
            return result
        lev=levels[k]; x=lev['omega']*lev['di']*b
        x+=lev['omega']*lev['di']*(b-lev['a']@x)
        x+=lev['p']@cycle(lev['r']@(b-lev['a']@x),k+1)
        for _ in range(2): x+=lev['omega']*lev['di']*(b-lev['a']@x)
        return x
    graph_input=cp.zeros(host.shape[0]); graph=None; graph_error=None
    # Captured temporary addresses must never be recycled into PCG state.
    # Keep all V-cycle allocations in a dedicated pool alive with the graph.
    graph_pool=cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(graph_pool.malloc):
        cycle(graph_input); cp.cuda.get_current_stream().synchronize()
    stream=cp.cuda.Stream(non_blocking=True)
    try:
        with stream,cp.cuda.using_allocator(graph_pool.malloc):
            stream.begin_capture(); graph_output=cycle(graph_input); graph=stream.end_capture()
    except Exception as error:
        graph_error=repr(error)
        try: stream.end_capture()
        except Exception: pass
    def precondition(r):
        if graph is None: return cycle(r)
        graph_input[:]=r
        graph.launch(stream=cp.cuda.get_current_stream())
        return graph_output.copy()
    backend=CuPyPreparedConjugateGradientBackend(host)
    report=dict(build_seconds=build_s,upload_seconds=upload_s,levels=[int(l['a'].shape[0]) for l in levels],
                capture_spmv=a.capture_spmv,spmv_max_abs_error=spmv_max_error,
                graph_workspace_pool_bytes=graph_pool.total_bytes(),
                operator_complexity=hierarchy.operator_complexity(),graph=graph is not None,graph_error=graph_error,
                hierarchy_device_bytes=sum(sum(x.nbytes for x in (l['a'].data,l['a'].indices,l['a'].indptr,l['di']))+
                 (sum(x.nbytes for key in ('p','r') for x in (l[key].data,l[key].indices,l[key].indptr)) if 'p' in l else 0) for l in levels)+inverse.nbytes,runs=[])
    for path in files:
        d=np.load(path); rhs,initial=d['rhs'],d['initial']
        control=backend.solve(rhs,initial,relative_tolerance=float(d['rtol']),absolute_tolerance=float(d['atol']),max_iterations=int(d['max_iterations']))
        b=cp.asarray(rhs); x=cp.asarray(initial); matrix=original_fine
        r=b-matrix@x; norm0=float(cp.linalg.norm(r).get())
        tol=float(d['rtol'])*norm0+float(d['atol']); cp.cuda.get_current_stream().synchronize()
        t=time.perf_counter(); z=precondition(r); direction=z.copy(); rz=cp.dot(r,z); converged=False
        for iteration in range(1,min(int(d['max_iterations']),500)+1):
            ad=matrix@direction; alpha=rz/cp.dot(direction,ad)
            x+=alpha*direction; r-=alpha*ad
            norm=float(cp.linalg.norm(r).get())
            if not np.isfinite(norm): raise RuntimeError('nonfinite residual')
            if norm<=tol: converged=True; break
            z=precondition(r); next_rz=cp.dot(r,z); direction=z+(next_rz/rz)*direction;rz=next_rz
        true_residual=float(cp.linalg.norm(b-matrix@x).get()); elapsed=time.perf_counter()-t
        delta=float(cp.max(cp.abs(x-cp.asarray(control['solution']))).get())
        controls=[]
        for _ in range(3):
            t=time.perf_counter(); baseline=backend.solve(rhs,initial,relative_tolerance=float(d['rtol']),absolute_tolerance=float(d['atol']),max_iterations=int(d['max_iterations']))
            controls.append(time.perf_counter()-t)
        row=dict(file=path.name,amg_s=elapsed,amg_iterations=iteration,cg_s=float(np.median(controls)),cg_iterations=baseline['iterations'],
                 tolerance=tol,true_residual=true_residual,converged=bool(converged and true_residual<=tol),potential_max_delta=delta)
        report['runs'].append(row);Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    print('AMG_SCREEN_COMPLETE',flush=True)

if __name__=='__main__':main()
