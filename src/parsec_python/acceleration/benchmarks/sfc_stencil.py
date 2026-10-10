"""Permutation/shared-halo ablation on unchanged FP64 sector stencil.

Retains every active unknown, every coefficient and the original CSR sum
order. No padding becomes a physical grid point. Times GPU action only;
construction cost and memory are reported separately, never hidden as SCF.
"""
import argparse
import gc
import json
import os
import socket
import time
from pathlib import Path
import numpy as np
import cupy as cp
from parsec_python.acceleration.backends.cupy_stencil_major import (
    StencilMajorHostMetadata, CuPyStencilMajorFiniteDifference)

def hilbert_keys(points):
    """Vectorized 3-D axes-to-transpose Hilbert mapping (Skilling transform)."""
    x = np.asarray(points, dtype=np.int64).copy()
    if x.ndim != 2 or x.shape[1] != 3 or np.any(x < 0):
        raise ValueError('nonnegative integer triples required')
    bits = max(1, int(x.max(initial=0)).bit_length())
    if bits > 21:
        raise ValueError('Hilbert keys exceed uint64')
    q = 1 << (bits-1)
    while q > 1:
        mask = q-1
        for axis in range(3):
            invert = (x[:, axis] & q) != 0
            t = (x[:, 0] ^ x[:, axis]) & mask
            t[invert] = 0
            x[:, 0] ^= np.where(invert, mask, t)
            x[:, axis] ^= t
        q >>= 1
    for axis in range(1, 3):
        x[:, axis] ^= x[:, axis-1]
    t = np.zeros(len(x), dtype=np.int64)
    q = 1 << (bits-1)
    while q > 1:
        t ^= np.where((x[:, 2] & q) != 0, q-1, 0)
        q >>= 1
    x ^= t[:, None]
    keys = np.zeros(len(x), dtype=np.uint64)
    for bit in range(bits-1, -1, -1):
        for axis in range(3):
            keys = (keys << np.uint64(1)) | ((x[:, axis] >> bit) & 1).astype(np.uint64)
    return keys

def permutation(xyz, mode):
    xyz = xyz - xyz.min(axis=0)
    if mode == 'original':
        return np.arange(len(xyz))
    if mode == 'point_hilbert':
        return np.argsort(hilbert_keys(xyz), kind='stable')
    blocks, inside = xyz // 2, xyz % 2
    local = inside[:, 0]*4 + inside[:, 1]*2 + inside[:, 2]
    if mode == 'block_hilbert':
        key = hilbert_keys(blocks)
    else:
        span = blocks.max(axis=0)+1
        key = (blocks[:, 0]*span[1] + blocks[:, 1])*span[2] + blocks[:, 2]
    return np.lexsort((local, key))

def permuted_metadata(data, order):
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    neighbors = data['neighbors'][:, order].copy()
    active = neighbors >= 0
    neighbors[active] = inverse[neighbors[active]]
    # Deliberately retain original slot order rather than re-sort CSR.
    return StencilMajorHostMetadata((len(order),)*2, neighbors,
                                   data['codes'][:, order], data['palette'])

SHARED_SOURCE = r'''
extern "C" __global__ void halo_stencil(
    int rows, int slots, int tile, int pitch,
    const int* offsets, const int* sources,
    const unsigned short* local, const unsigned char* codes,
    const double* palette, const int* output_rows, const double* x, double* y, int width) {
    extern __shared__ double values[];
    int block = blockIdx.x;
    int begin = offsets[block], count = offsets[block+1]-begin;
    for (int index=threadIdx.x; index<count*width; index+=blockDim.x) {
        int column=index/count, halo=index-column*count;
        values[index]=x[(long long)column*rows+sources[begin+halo]];
    }
    __syncthreads();
    int row=block*tile+threadIdx.x;
    if (threadIdx.x>=tile || row>=rows) return;
    double a[6]={0,0,0,0,0,0};
    for (int s=0;s<slots;++s) {
        long long offset=(long long)s*rows+row;
        unsigned short k=local[offset];
        if(k==65535) continue;
        double c=palette[codes[offset]];
        #pragma unroll
        for(int j=0;j<6;++j) if(j<width) a[j]+=c*values[j*count+k];
    }
    #pragma unroll
    for(int j=0;j<6;++j) if(j<width) y[(long long)j*rows+output_rows[row]]=a[j];
}
'''

def shared_operator(metadata, tile, output_rows=None):
    neighbors = metadata.neighbors
    rows = neighbors.shape[1]
    row_map = np.arange(rows, dtype=np.int32) if output_rows is None else np.asarray(output_rows, dtype=np.int32)
    offsets = [0]
    sources = []
    local = np.full(neighbors.shape, 65535, dtype=np.uint16)
    counts = []
    for start in range(0, rows, tile):
        stop = min(start+tile, rows)
        chunk = neighbors[:, start:stop]
        mask = chunk >= 0
        unique, inverse = np.unique(row_map[chunk[mask]], return_inverse=True)
        if len(unique) >= 65535:
            raise ValueError('halo index overflow')
        local[:, start:stop][mask] = inverse.astype(np.uint16)
        sources.append(unique)
        offsets.append(offsets[-1]+len(unique))
        counts.append(len(unique))
    halo = max(counts)
    shared_width = min(6, 48000//(halo*8))
    if shared_width < 1:
        raise ValueError('halo exceeds conservative shared-memory limit')
    buffers = [cp.asarray(np.asarray(offsets, dtype=np.int32)),
               cp.asarray(np.concatenate(sources).astype(np.int32)), cp.asarray(local),
               cp.asarray(metadata.coefficient_codes), cp.asarray(metadata.coefficient_palette), cp.asarray(row_map)]
    kernel = cp.RawKernel(SHARED_SOURCE, 'halo_stencil', options=('--std=c++11',))
    kernel.compile()
    def apply(x, out):
        for start in range(0, x.shape[1], shared_width):
            width = min(shared_width, x.shape[1]-start)
            kernel((len(counts),), (max(64, tile),),
                   (np.int32(rows), np.int32(neighbors.shape[0]), np.int32(tile), np.int32(halo),
                    *buffers, x[:, start:], out[:, start:], np.int32(width)),
                   shared_mem=halo*width*8)
        return out
    return apply, dict(tile=tile, max_halo=halo, mean_halo=float(np.mean(counts)),
                       width=shared_width, metadata_bytes=sum(v.nbytes for v in buffers))

def measure(operation, x, out, repeats=21):
    # CPU metadata construction can let clocks drop. Warm for wall time,
    # then time batches to reduce launch/event overhead on sub-ms kernels.
    until=time.perf_counter()+0.25
    while time.perf_counter()<until:
        for _ in range(10): operation(x,out)
        cp.cuda.get_current_stream().synchronize()
    times = []
    for _ in range(repeats):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(10): operation(x,out)
        stop.record()
        stop.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(start, stop))/10)
    return dict(median_ms=float(np.median(times)), min_ms=min(times), samples_ms=times, operations_per_sample=10)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    device_name=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    if isinstance(device_name,bytes): device_name=device_name.decode('ascii')
    provenance=dict(node=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),
                    device=str(device_name),cupy_version=cp.__version__,
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
    # A complete cube must be visited exactly once with face-adjacent steps.
    for side in (2,4,8):
        cube=np.indices((side,)*3).reshape(3,-1).T
        keys=hilbert_keys(cube)
        assert np.array_equal(np.sort(keys), np.arange(side**3))
        path=cube[np.argsort(keys)]
        assert np.all(np.abs(np.diff(path,axis=0)).sum(axis=1)==1)
    with np.load(args.input) as archive:
        data={key:archive[key] for key in archive.files}
    xyz=data['coordinates']
    n=len(xyz)
    host=np.random.default_rng(72031).standard_normal((n,6))
    records=[]
    reference=None
    for mode in ('original','blocked_cartesian','point_hilbert','block_hilbert'):
        began=time.perf_counter()
        order=permutation(xyz,mode)
        assert np.unique(order).size==n
        metadata=permuted_metadata(data,order)
        setup=time.perf_counter()-began
        x=cp.asarray(host[order],order='F')
        out=cp.empty_like(x)
        standard=CuPyStencilMajorFiniteDifference(cp,metadata=metadata)
        standard.apply(x,output=out)
        actual=cp.asnumpy(out)[np.argsort(order)]
        if reference is None: reference=actual
        error=float(np.max(np.abs(actual-reference)))
        assert error < 2e-12, error
        measured=measure(lambda a,b: standard.apply(a,output=b),x,out)
        records.append(dict(mode=mode,kernel='global',rows=n,provenance=provenance,permutation_seconds=setup,
                            max_abs_error=error,metadata_bytes=metadata.neighbors.nbytes+
                            metadata.coefficient_codes.nbytes+metadata.coefficient_palette.nbytes,**measured))
        for layout in ('permuted', 'original'):
            # Original storage only changes the kernel's spatial scheduling;
            # wavefunctions, KB factors, potentials and all external maps stay
            # in their existing layout. This tests a narrowly integrable path.
            if layout == 'original':
                x=cp.asarray(host,order='F')
            for tile in (64,128,256):
                began=time.perf_counter()
                apply,info=shared_operator(metadata,tile,order if layout=='original' else None)
                setup_shared=time.perf_counter()-began
                apply(x,out)
                actual=cp.asnumpy(out)
                if layout=='permuted': actual=actual[np.argsort(order)]
                error=float(np.max(np.abs(actual-reference)))
                assert error < 2e-12, error
                measured=measure(apply,x,out)
                records.append(dict(mode=mode,kernel='shared_halo',storage=layout,rows=n,provenance=provenance,
                                    permutation_seconds=setup,halo_setup_seconds=setup_shared,
                                    max_abs_error=error,**info,**measured))
                del apply
        del standard,x,out,metadata
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        args.output.write_text(json.dumps(records,indent=2))
        print('SFC_MODE_COMPLETE',mode,[(r['kernel'],r.get('tile'),r['median_ms']) for r in records if r['mode']==mode],flush=True)

if __name__=='__main__': main()
