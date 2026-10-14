"""Exact operator compression screening on archived production metadata."""
import argparse, gc, json, os, time
from pathlib import Path
import numpy as np
import cupy as cp
from parsec_python.acceleration.backends.cupy_stencil_major import StencilMajorHostMetadata, CuPyStencilMajorFiniteDifference
from parsec_python.acceleration.benchmarks.sfc_stencil import measure

def main():
    p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output'); args=p.parse_args()
    with np.load(args.input) as d:
        m=StencilMajorHostMetadata((d['neighbors'].shape[1],)*2,d['neighbors'],d['codes'],d['palette'])
    rows=m.shape[0]; rng=np.random.default_rng(711)
    x=cp.array(rng.normal(size=(rows,6)),order='F'); out=cp.empty_like(x)
    potential=cp.asarray(rng.normal(size=rows)); previous=cp.array(rng.normal(size=(rows,6)),order='F')
    results=[]
    for tile in (0,16,32,64,128,256):
        os.environ['PARSEC_CUPY_IMPLICIT_TILE']=str(tile)
        t=time.perf_counter(); op=CuPyStencilMajorFiniteDifference(cp,metadata=m)
        cp.cuda.get_current_stream().synchronize(); setup=time.perf_counter()-t
        def action(a,b): return op.apply(a,potential,output=b)
        action(x,out)
        if tile==0: reference=out.copy()
        error=float(cp.max(cp.abs(out-reference)).get())
        timings=measure(action,x,out)
        op.chebyshev_recurrence(x,potential,previous=previous,center=1.3,scale=.08,sigma=.9,sigma_next=.83,output=out)
        if tile==0: recurrence_ref=out.copy()
        recurrence_error=float(cp.max(cp.abs(out-recurrence_ref)).get())
        item=dict(tile=tile,setup_s=setup,metadata_bytes=sum(a.nbytes for a in (op.neighbors,op.coefficient_codes,op.coefficient_palette)),
                  max_error=error,recurrence_error=recurrence_error,**timings,
                  compression=getattr(op,'implicit_statistics',None))
        results.append(item); print(json.dumps(item),flush=True)
        Path(args.output).write_text(json.dumps(results,indent=2))
        if error>2e-12 or recurrence_error>2e-12: raise RuntimeError('operator accuracy gate failed')
        del op; gc.collect(); cp.get_default_memory_pool().free_all_blocks()

if __name__=='__main__': main()
