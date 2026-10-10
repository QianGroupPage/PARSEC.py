"""Orbital-block distribution inside a symmetry sector.

:class:`DistributedFilter` holds what the devices of one sector filter with:
a replica of the sector operator on every device but the owner, a stream and
the filter graphs of each.  The shared basis of :mod:`distributed_state`,
which a sector takes by default where
:func:`distributed_state.shared_basis_devices` allows it, uses it together
with :func:`_pool`, :func:`block_partitions` and :func:`starting_sigmas`.

:func:`distributed_filter` is the opt-in route of
``PARSEC_CUPY_DISTRIBUTED_FILTER=1``.  It keeps the complete coupled Ritz
solve on the owning device.  Every transfer, replica build and gather is part
of the caller's elapsed time.  It does not distribute the Ritz step or remove
the memory bottleneck of the sector owner.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from time import perf_counter
from weakref import ref
import os
import numpy as np
import scipy.sparse as sp
from ..backends.cupy import require_cupy, CuPyHamiltonian
from ..backends.cupy_stencil_major import StencilMajorHostMetadata
from ..backends.implicit_stencil import PackedTileHostMetadata
from .filter_graph import BlockFilterGraphs
from .chebyshev import FilterBlock

_POOLS={}
_LOCK=Lock()


def _pool(device):
    with _LOCK:
        if device not in _POOLS:
            _POOLS[device]=ThreadPoolExecutor(max_workers=1,thread_name_prefix=f"orbital-shard-{device}")
        return _POOLS[device]


def block_partitions(blocks, devices):
    """Contiguous partitions balanced by degree times orbital count."""
    weights=np.array([(b.stop-b.start)*b.degree for b in blocks])
    cumulative=np.cumsum(weights); total=int(cumulative[-1])
    cuts=[0]
    for k in range(1,min(devices,len(blocks))):
        cut=int(np.searchsorted(cumulative,total*k/min(devices,len(blocks))))+1
        cut=max(cuts[-1]+1,min(cut,len(blocks)-(min(devices,len(blocks))-k)))
        cuts.append(cut)
    cuts.append(len(blocks))
    return tuple((a,b) for a,b in zip(cuts,cuts[1:]))


def starting_sigmas(blocks,lower,upper,reset,reference=None):
    """Sigma carried into each block; ``reference`` defaults to the lower bound."""
    half=.5*(float(upper)-float(lower)); center=.5*(float(upper)+float(lower))
    sigma_one=half/(float(lower if reference is None else reference)-center)
    carried=None; starts=[]
    for block in blocks:
        starts.append(carried)
        sigma=sigma_one if reset or carried is None else carried
        for _ in range(1,block.degree): sigma=1/(2/sigma_one-sigma)
        carried=sigma
    return starts


def replica_stencil(operator):
    """Host metadata from which another device builds the stencil of ``operator``, and its tile size.

    A replica holds the stencil in the layout of the owner; the tile size is
    0 for the slot-major one. Affine tiles are packed once, by the owner: its
    packed arrays are downloaded as they are, and no replica packs.
    """
    cp,_=require_cupy()
    stencil=operator.compact_finite_difference
    statistics=getattr(stencil,'implicit_statistics',None)
    tile=int(statistics['tile']) if statistics else 0
    with cp.cuda.Device(int(operator.effective_potential.device.id)):
        arrays=cp.asnumpy(stencil.neighbors),cp.asnumpy(stencil.coefficient_codes),cp.asnumpy(stencil.coefficient_palette)
    if tile:
        return PackedTileHostMetadata(operator.shape,stencil.slot_count,*arrays,dict(statistics)),tile
    return StencilMajorHostMetadata(operator.shape,*arrays),0


class DistributedFilter:
    def __init__(self,operator,devices):
        cp,_=require_cupy()
        self.owner=int(operator.effective_potential.device.id)
        self.owner_ref=ref(operator);self.devices=devices;self.replicas={};self.streams={};self.graphs={}
        started=perf_counter()
        metadata,tile=replica_stencil(operator)
        with cp.cuda.Device(self.owner):
            offsets,columns,values=operator.projector_csr_data
            projectors=sp.csr_matrix((cp.asnumpy(values),cp.asnumpy(columns),cp.asnumpy(offsets)),
                                    shape=(operator.shape[0],operator.projector_count))
            signs=cp.asnumpy(operator.projector_signs)
        for device in devices:
            with cp.cuda.Device(device):
                self.streams[device]=cp.cuda.Stream(non_blocking=True)
                if device!=self.owner:
                    self.replicas[device]=CuPyHamiltonian(None,np.zeros(operator.shape[0]),(projectors,signs),
                        retain_generic_laplacian=False,finite_difference_metadata=metadata,implicit_tile=tile)
                self.graphs[device]=BlockFilterGraphs(operator if device==self.owner else self.replicas[device])
        # Wall time of the downloads from the owner and of the replicas, for reports.
        self.replica_seconds=perf_counter()-started

    def apply(self,matrix,blocks,lower,upper,reset,out=None):
        cp,_=require_cupy();operator=self.owner_ref()
        if operator is None:raise ReferenceError('filter owner released')
        # The old orbital matrix and potential are fully produced before any
        # worker reads them on another stream/device.
        source=matrix;matrix=cp.asfortranarray(matrix)
        potential=cp.asnumpy(operator.effective_potential)
        cp.cuda.get_current_stream().synchronize()
        # A caller that owns the input may ask for the result in place. The
        # devices work on disjoint column ranges, so the owner needs no second
        # N-by-states array; helper copies are private and always filtered in place.
        in_place=out is not None and out is source and matrix is source
        output=matrix if in_place else cp.empty_like(matrix,order='F')
        partitions=block_partitions(blocks,len(self.devices));sigmas=starting_sigmas(blocks,lower,upper,reset)
        def run(device,first,last):
            begin,end=blocks[first].start,blocks[last-1].stop
            local_blocks=tuple(FilterBlock(b.start-begin,b.stop-begin,b.degree) for b in blocks[first:last])
            with cp.cuda.Device(device),self.streams[device]:
                if device==self.owner:
                    local=matrix[:,begin:end];target=local if in_place else None
                else:
                    self.replicas[device].effective_potential.set(potential,stream=self.streams[device])
                    local=cp.array(matrix[:,begin:end],order='F',copy=True);target=local
                result=self.graphs[device].apply(local,local_blocks,lower,upper,lower,reset,initial_sigma=sigmas[first],out=target)
                self.streams[device].synchronize()
                return begin,end,result
        futures=[_pool(device).submit(run,device,first,last)
                 for device,(first,last) in zip(self.devices,partitions)]
        wait(futures)
        with cp.cuda.Device(self.owner):
            stream=cp.cuda.get_current_stream()
            for future in futures:
                begin,end,result=future.result()
                target=output[:,begin:end]
                if int(result.device.id)!=self.owner:
                    # Both sides are contiguous column ranges: copy device to
                    # device straight into place, without an owner-side temporary.
                    target.data.copy_from_device_async(cp.asfortranarray(result).data,result.nbytes,stream)
                elif not in_place:
                    target[...]=result
            stream.synchronize()
        return output


def distributed_filter(operator,matrix,blocks,lower,upper,reset,out=None):
    cp,_=require_cupy()
    # MPI sector owners may each have a disjoint subset of this node's
    # devices. Never let independent sectors commandeer the other group.
    assigned=getattr(operator,'distributed_filter_devices',None)
    if assigned is None:
        raw=os.environ.get('PARSEC_CUPY_DEVICES','auto').strip().lower()
        if raw in {'current','off'}:
            devices=(int(operator.effective_potential.device.id),)
        elif raw in {'auto',''}:
            devices=tuple(range(cp.cuda.runtime.getDeviceCount()))
        else:
            devices=tuple(map(int,raw.split(',')))
    else:
        devices=tuple(assigned)
    if not devices or len(set(devices))!=len(devices):
        raise ValueError('distributed filter device group must be nonempty and unique')
    if int(operator.effective_potential.device.id) not in devices:
        raise ValueError('distributed filter device group must include its owner')
    if len(devices)==1:
        graph=getattr(operator,'_filter_graph_workspace',None)
        if graph is None:
            graph=BlockFilterGraphs(operator);operator._filter_graph_workspace=graph
        return graph.apply(matrix,blocks,lower,upper,lower,reset,out=out)
    worker=getattr(operator,'_distributed_filter',None)
    if worker is None:
        worker=DistributedFilter(operator,devices);operator._distributed_filter=worker
    return worker.apply(matrix,blocks,lower,upper,reset,out=out)
