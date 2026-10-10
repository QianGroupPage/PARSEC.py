"""Optional same-stream NCCL sums; point-to-point halos still use MPI.

This mixin is only for the experimental CUDA replay, never a production default.
The caller must launch every collective in identical order and abort on failure.
"""
import os


class NcclReductionMixin:
    def __init__(self, *args, **kwargs):
        self._nccl = None
        super().__init__(*args, **kwargs)
        if not self.gpu or self.transport != 'cuda':
            raise ValueError('NCCL sums require the CUDA transport and CuPy')
        from cupy.cuda import nccl
        self._nccl_module = nccl
        identifier = self.comm.bcast(nccl.get_unique_id() if self.rank == 0 else None, root=0)
        self._nccl = nccl.NcclCommunicator(self.size, identifier, self.rank)
        self.reduction_metadata = {
            'backend': 'nccl', 'version': nccl.get_version(),
            'environment': {key: value for key, value in os.environ.items()
                            if key.startswith(('NCCL_', 'FI_CXI_'))},
        }

    def _sum(self, values):
        if self._nccl is None:
            return super()._sum(values)
        if values.dtype != self.xp.dtype('float64'):
            raise ValueError('only FP64 reductions are supported')
        packed = self.xp.ascontiguousarray(values)
        result = self.xp.empty_like(packed)
        if packed.size:
            self._nccl.allReduce(packed.data.ptr, result.data.ptr, packed.size,
                                 self._nccl_module.NCCL_FLOAT64,
                                 self._nccl_module.NCCL_SUM,
                                 self.xp.cuda.get_current_stream().ptr)
        # Producers, reduction and consumers share a CUDA stream. MPI halo
        # packing explicitly synchronizes it; benchmark timers do so as well.
        return result

    def close(self):
        if self._nccl is not None:
            self._sync()
            self._nccl.destroy()
            self._nccl = None
        return super().close()
