"""Exact Galerkin aggregation screen using verified production grid rows."""
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from ..backends.cupy_stencil_major import StencilMajorHostMetadata


def hierarchy_from_capture(matrix,path,max_coarse=96):
    with np.load(path) as d:
        metadata=StencilMajorHostMetadata(matrix.shape,d['neighbors'],d['codes'],d['palette'])
        kinetic=metadata.to_csr();coords=d['coordinates'].copy()
    # Shape alone cannot establish row identity. Check the full captured
    # kinetic operator against the Poisson matrix before using its coordinates.
    if not (np.array_equal(matrix.indptr,kinetic.indptr)
            and np.array_equal(matrix.indices,kinetic.indices)
            and np.array_equal(matrix.data,kinetic.data)):
        raise ValueError('captured coordinate row order/operator is not identical to Poisson')
    levels=[];original_nnz=matrix.nnz
    while matrix.shape[0]>max_coarse:
        coarse,inverse=np.unique(coords//2,axis=0,return_inverse=True)
        if len(coarse)==matrix.shape[0]:raise ValueError('geometric coarsening stalled')
        prolong=sp.csr_matrix((np.ones(len(coords)),(np.arange(len(coords)),inverse)),shape=(len(coords),len(coarse)))
        restrict=prolong.T.tocsr()
        levels.append(SimpleNamespace(A=matrix,P=prolong,R=restrict))
        matrix=(restrict@matrix@prolong).tocsr();matrix.sum_duplicates();matrix.sort_indices();coords=coarse
    levels.append(SimpleNamespace(A=matrix))
    return SimpleNamespace(levels=levels,operator_complexity=lambda:sum(l.A.nnz for l in levels)/original_nnz)
