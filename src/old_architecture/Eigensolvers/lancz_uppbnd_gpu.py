import cupy as cp

def lancz_uppbnd(n, A, k=6):
    k = min(max(k, 6), 10)

    # Tridiagonal and initial vector
    T = cp.zeros((k, k), dtype=cp.float32)
    v = cp.random.rand(n, dtype=cp.float32)
    v /= cp.linalg.norm(v)

    tol = 2.5e-16
    upperb = cp.zeros((3, k), dtype=cp.float32)

    f = A @ v
    alpha = cp.dot(v, f)
    f = f - alpha * v
    T[0, 0] = alpha
    beta = cp.linalg.norm(f)

    upperb[:, 0] = alpha + beta

    ritzv = None
    isbreak = False

    for j in range(1, k):
        if beta > tol:
            v0 = v
            v = f / beta
            f = A @ v
            f = f - beta * v0
            alpha = cp.dot(v, f)
            f = f - alpha * v
            T[j, j - 1] = beta
            T[j - 1, j] = beta
            T[j, j] = alpha
        else:
            isbreak = True
            break

        beta = cp.linalg.norm(f)

        if not isbreak:
            ritzv, _ = cp.linalg.eigh(T[:j, :j])
        else:
            ritzv, _ = cp.linalg.eigh(T[:j - 1, :j - 1])

        if beta < 1e-2:
            beta *= 10

        upperb[0, j] = ritzv[-1] + beta
        upperb[1, j] = ritzv[-1] + beta
        upperb[2, j] = ritzv[-1] + beta

    uppbnd = (upperb[0, j] + upperb[1, j]) / 2
    uppbnd = (upperb[2, j] + uppbnd) / 2

    return uppbnd, ritzv
