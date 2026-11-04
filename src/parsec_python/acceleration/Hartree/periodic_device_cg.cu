// Fused conjugate gradients of the periodic, wire and slab Hartree solves (periodic_device.py).
//
// One CG iteration of parsec_python.Hartree.poisson._conjugate_gradient is
//   Ap = A p;  den = p.Ap;  alpha = rr/den;  x += alpha p;  r -= alpha Ap;
//   rr' = r.r; beta = rr'/rr;  p = r + beta p.
// These kernels run a batch of iterations without a host synchronization:
// every scalar stays on the device (history arrays), the directions p_i are
// kept so that x is updated afterwards with exactly the iterations the host
// accepts (the stopping test and the breakdown rule of the reference).
// Dot products are block partial sums in a fixed order followed by one
// fixed-order block reduction: deterministic.  The CSR product sums each row
// in its index order, as SciPy's csr_matvec does.  Compiled with --fmad=false.

#define BLOCK 256

__device__ __forceinline__ double block_sum(double value, double *shared)
{
    const int t = threadIdx.x;
    shared[t] = value;
    __syncthreads();
    for (int width = BLOCK / 2; width > 0; width >>= 1) {
        if (t < width) shared[t] += shared[t + width];
        __syncthreads();
    }
    return shared[0];
}

// Ap = A p (CSR) and partial[block] = sum over the block of p_i (Ap)_i.
extern "C" __global__ void cg_spmv_dot(const int *indptr, const int *indices, const double *data,
                                       const double *p, double *Ap, double *partial, const long n)
{
    __shared__ double shared[BLOCK];
    const long i = (long)blockIdx.x * BLOCK + threadIdx.x;
    double product = 0.0;
    if (i < n) {
        double sum = 0.0;
        for (int j = indptr[i]; j < indptr[i + 1]; ++j) sum += data[j] * p[indices[j]];
        Ap[i] = sum;
        product = p[i] * sum;
    }
    const double total = block_sum(product, shared);
    if (threadIdx.x == 0) partial[blockIdx.x] = total;
}

// out[slot] = sum of partial[0..count) in a fixed order (one block).
extern "C" __global__ void cg_reduce(const double *partial, const long count, double *out, const int slot)
{
    __shared__ double shared[BLOCK];
    double sum = 0.0;
    for (long k = threadIdx.x; k < count; k += BLOCK) sum += partial[k];
    const double total = block_sum(sum, shared);
    if (threadIdx.x == 0) out[slot] = total;
}

// alpha = rr[i]/den[i];  r -= alpha Ap;  partial[block] = sum r^2.
// history: den[0..m), rr[0..m] with rr[0] the residual before the batch.
extern "C" __global__ void cg_update_residual(double *r, const double *Ap, const double *den,
                                              const double *rr, const int i, double *partial, const long n)
{
    __shared__ double shared[BLOCK];
    const long k = (long)blockIdx.x * BLOCK + threadIdx.x;
    const double alpha = rr[i] / den[i];
    double square = 0.0;
    if (k < n) {
        const double value = r[k] - alpha * Ap[k];
        r[k] = value;
        square = value * value;
    }
    const double total = block_sum(square, shared);
    if (threadIdx.x == 0) partial[blockIdx.x] = total;
}

// stored = p;  beta = rr[i+1]/rr[i];  p = r + beta p.
extern "C" __global__ void cg_update_direction(double *p, const double *r, double *stored,
                                               const double *rr, const int i, const long n)
{
    const long k = (long)blockIdx.x * BLOCK + threadIdx.x;
    if (k < n) {
        const double beta = rr[i + 1] / rr[i];
        const double old = p[k];
        stored[k] = old;
        p[k] = r[k] + beta * old;
    }
}

// x += alpha_j p_j for j = 0..accepted-1 in order, alpha_j = rr[j]/den[j].
extern "C" __global__ void cg_apply_solution(double *x, const double *stored, const double *den,
                                             const double *rr, const int accepted, const long n)
{
    const long k = (long)blockIdx.x * BLOCK + threadIdx.x;
    if (k < n) {
        double value = x[k];
        for (int j = 0; j < accepted; ++j) {
            const double alpha = rr[j] / den[j];
            value += alpha * stored[(long)j * n + k];
        }
        x[k] = value;
    }
}
