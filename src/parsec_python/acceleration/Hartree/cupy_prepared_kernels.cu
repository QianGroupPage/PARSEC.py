// FP64 CG, fixed-order reductions. A stopped recurrence is immutable inside
// captured graph groups. Initial/final residual matvecs explicitly bypass it.
// NVRTC provides device math builtins; no host C/C++ headers are needed.
typedef COEFF_CODE code_t;

__device__ double block_sum(double value) {
    __shared__ double partial[256];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int offset = 128; offset; offset >>= 1) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    return partial[0];
}

__device__ double sum_partials(const double* partial, int count) {
    double value = 0.0;
    for (int i = threadIdx.x; i < count; i += 256) value += partial[i];
    return block_sum(value);
}

extern "C" __global__ void matvec(int n, int width, const int* neighbors,
    const code_t* codes, const double* palette, const double* x, double* ax,
    double* partial, const double* state, int force) {
    if (!force && state[4] != 0.0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0, dot = 0.0;
    if (i < n) {
        for (int slot = 0; slot < width; ++slot) {
            long long at = (long long)slot * n + i;
            int j = neighbors[at];
            if (j >= 0) value += palette[codes[at]] * x[j];
        }
        ax[i] = value;
        dot = x[i] * value;
    }
    double sum = block_sum(dot);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

extern "C" __global__ void residual(int n, const double* b, const double* ax,
    double* r, double* p, double* partial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (i < n) { value = b[i] - ax[i]; r[i] = value; p[i] = value; }
    double sum = block_sum(value * value);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

extern "C" __global__ void initialize(const double* partial, int count,
    double* state, double rtol, double atol) {
    double rr = sum_partials(partial, count);
    if (threadIdx.x == 0) {
        double norm = sqrt(rr), tol = rtol * norm + atol;
        state[0] = rr; state[1] = 0; state[2] = 0; state[3] = tol;
        state[4] = norm <= tol ? 1 : 0;
        state[5] = 0; state[6] = 1; state[7] = norm; state[8] = norm;
    }
}

extern "C" __global__ void alpha(const double* partial, int count, double* state) {
    if (state[4] != 0) return;
    double denominator = sum_partials(partial, count);
    if (threadIdx.x == 0) {
        state[6] += 1;
        if (!(denominator > 0) || !isfinite(denominator)) state[4] = -1;
        else state[1] = state[0] / denominator;
    }
}

extern "C" __global__ void update_xr(int n, double* x, double* r, const double* p,
    const double* ap, double* partial, const double* state) {
    if (state[4] != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0;
    if (i < n) { x[i] += state[1] * p[i]; r[i] -= state[1] * ap[i]; value = r[i]; }
    double sum = block_sum(value * value);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

extern "C" __global__ void beta(const double* partial, int count, double* state) {
    if (state[4] != 0) return;
    double rr = sum_partials(partial, count);
    if (threadIdx.x == 0) {
        state[5] += 1;
        state[2] = rr / state[0]; state[0] = rr; state[8] = sqrt(rr);
        if (state[8] <= state[3]) state[4] = 1;
    }
}

extern "C" __global__ void update_p(int n, const double* r, double* p, const double* state) {
    if (state[4] != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + state[2] * p[i];
}

extern "C" __global__ void finish(const double* partial, int count, double* state) {
    double rr = sum_partials(partial, count);
    if (threadIdx.x == 0) { state[8] = sqrt(rr); state[6] += 1; }
}
