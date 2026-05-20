import os
import time
import logging
import numpy as np
from numpy.linalg import svd as _full_svd
from scipy import sparse
from scipy.sparse.linalg import svds as _scipy_svds
from sklearn.utils.extmath import randomized_svd as _randomized_svd

logger = logging.getLogger(__name__)

try:
    import ray
    _HAS_RAY = True
except ImportError:
    ray = None
    _HAS_RAY = False


_SVD_SOLVERS = ("auto", "full", "randomized", "lanczos", "shared")


def _truncated_svd(M, nComp, solver, random_state, n_oversamples):
    """Return V (right singular vectors), shape (M.shape[1], nComp).

    pcCoefficients only needs the top-nComp right singular vectors of M; U and
    s are derivable from score = M @ V if the caller wants them. Returning just
    V keeps each backend's output shape uniform and minimises copies.
    """
    if solver == "full":
        _, _, VT = _full_svd(M, full_matrices=False)
        return VT[:nComp, :].T
    if solver == "randomized":
        # randomized_svd handles oversampling and power iterations internally.
        # Deterministic given random_state. Faster than full SVD when nComp << min(M.shape).
        _, _, VT = _randomized_svd(
            M,
            n_components=nComp,
            n_oversamples=n_oversamples,
            random_state=random_state,
        )
        return VT.T
    if solver == "lanczos":
        # scipy.sparse.linalg.svds requires k < min(M.shape); falls back to full
        # for tiny matrices where the constraint isn't met.
        k = nComp
        if k >= min(M.shape):
            _, _, VT = _full_svd(M, full_matrices=False)
            return VT[:nComp, :].T
        _, s, VT = _scipy_svds(M, k=k)
        # svds returns ascending singular values; reorder to descending.
        order = np.argsort(s)[::-1]
        return VT[order, :].T
    raise ValueError(f"unknown svd_solver={solver!r}; expected one of {_SVD_SOLVERS}")


def _resolve_solver(solver, m, n_minus_1, nComp):
    """Pick a concrete solver when caller passes 'auto'.

    Heuristic: for small problems the constant-factor overhead of randomized
    SVD beats its asymptotic win, so use full SVD. The crossover is empirical
    but ~n*m > 5e4 holds in practice on modern CPUs.
    """
    if solver != "auto":
        return solver
    if n_minus_1 <= max(50, nComp * 5):
        return "full"
    if m * n_minus_1 < 5_000:
        return "full"
    return "randomized"


def pcCoefficients(X, K, nComp, svd_solver="full", random_state=0, n_oversamples=10):
    """Top-nComp PC regression of column K of X on the remaining columns.

    Returns a length-(n-1) coefficient vector. Sign-invariant w.r.t. singular
    vectors, so randomized solvers reproduce the full-SVD result to working
    precision.
    """
    y = X[:, K]
    Xi = np.delete(X, K, 1)
    V = _truncated_svd(Xi, nComp, svd_solver, random_state, n_oversamples)
    score = Xi @ V                      # (cells, nComp)
    t = np.sqrt(np.sum(score**2, axis=0))
    score_lsq = score / (t**2)[None, :]
    beta_pc = score_lsq.T @ y           # (nComp,)
    return V @ beta_pc                  # (n-1,)


def _build_B_local(X, indices, nComp, svd_solver, random_state, n_oversamples):
    """Compute coefficient rows for the given gene indices in-process."""
    out = np.empty((len(indices), X.shape[1] - 1), dtype=float)
    for i, k in enumerate(indices):
        out[i] = pcCoefficients(
            X, k, nComp,
            svd_solver=svd_solver,
            random_state=random_state,
            n_oversamples=n_oversamples,
        )
    return out


def _build_B_shared(X, nComp, random_state, n_oversamples, n_iter=None):
    """Compute the full coefficient matrix via a single sketched range-finder
    of X, reusing it across all genes.

    Mathematical equivalence: for each gene j, the per-gene path computes
    top-nComp right singular vectors of Xi = X[:, ≠j]. Here we approximate
    Xi ≈ Q (Qᵀ Xi) where Q is a randomized range basis for X (range(Xi) ⊆
    range(X)). The shared range Q lets us compute Xi's top-k SVD via a small
    k_total × (n-1) SVD per gene, replacing the m × (n-1) SVD per gene.

    Approximation error is bounded by the residual ||(I - QQᵀ) X|| which the
    range-finder with power iterations makes ~σ_{k+1}. For low-rank scRNA-seq
    spectra this matches the per-gene randomized solver within float tolerance.
    """
    m, n = X.shape
    # shared SVD needs Q to cover all of X's significant directions in one
    # sketch (not just Xi's, as per-gene randomized would). Bump oversampling
    # accordingly so a single Q can serve every leave-one-out problem.
    n_oversamples = max(n_oversamples, 20)
    k_total = nComp + n_oversamples
    if k_total >= min(m, n):
        # Fall back gracefully on tiny problems where the sketch can't fit.
        n_oversamples = max(0, min(m, n) - nComp - 1)
        k_total = nComp + n_oversamples
        if k_total < nComp + 1:
            raise ValueError(
                f"shared solver needs nComp+1 < min(m, n); got nComp={nComp}, "
                f"min(m,n)={min(m, n)}"
            )
    rng = np.random.default_rng(random_state)
    Omega = rng.standard_normal((n, k_total))
    Y = X @ Omega                              # m × k_total
    if n_iter is None:
        n_iter = 4 if k_total <= 0.1 * min(m, n) else 7
    XT = X.T
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y)
        Y = X @ (XT @ Q)
    Q, _ = np.linalg.qr(Y)                     # m × k_total, orthonormal cols
    C = Q.T @ X                                # k_total × n

    n_minus_1 = n - 1
    B = np.empty((n, n_minus_1), dtype=float)
    # Precompute mask templates for fast column removal
    for j in range(n):
        # C_j is C with column j removed.
        # np.delete copies; for k_total × n with k_total small this is cheap.
        C_j = np.delete(C, j, axis=1)          # k_total × (n-1)
        _, _, VT_j = _full_svd(C_j, full_matrices=False)
        V_k = VT_j[:nComp, :].T                # (n-1) × nComp

        # In Q-space: Xi ≈ Q @ C_j, so Xi @ V_k ≈ Q @ (C_j @ V_k).
        # Q is orthonormal → ||Q @ Sk||² = ||Sk||², and (Xi @ V_k)ᵀ y = Skᵀ (Qᵀ y).
        Sk = C_j @ V_k                          # k_total × nComp
        Qty = C[:, j]                           # k_total  (= Qᵀ X[:, j])
        score_norm2 = (Sk * Sk).sum(axis=0)
        beta_pc = (Sk.T @ Qty) / score_norm2
        B[j] = V_k @ beta_pc
    return B


def _assemble_A(B, n, scale, symmetric, q, as_sparse):
    A = np.zeros((n, n), dtype=float)
    A[~np.eye(n, dtype=bool)] = B.ravel()
    if scale:
        A = A / np.abs(A).max()
    if q > 0:
        A[np.abs(A) < np.percentile(np.abs(A), q)] = 0
    if symmetric:
        A = (A + A.T) / 2
    if as_sparse:
        A = sparse.csc_matrix(A)
    return A


def pcNet(X,
          nComp: int = 3,
          scale: bool = True,
          symmetric: bool = True,
          q: float = 0.,
          as_sparse: bool = True,
          random_state: int = 0,
          svd_solver: str = "auto",
          n_oversamples: int = 10):
    """Build the principal-component regression network from X (cells x genes).

    svd_solver:
        - "auto"      : pick "full" for small inputs, "randomized" otherwise
        - "full"      : numpy economy SVD (legacy; numerically reference)
        - "randomized": sklearn randomized_svd (deterministic, big speedup)
        - "lanczos"   : scipy svds (ARPACK)
    """
    if svd_solver not in _SVD_SOLVERS:
        raise ValueError(f"svd_solver={svd_solver!r}; expected one of {_SVD_SOLVERS}")
    X = X.toarray() if sparse.issparse(X) else X
    if nComp < 2 or nComp >= X.shape[1]:
        raise ValueError("nComp should be greater or equal than 2 and lower than the total number of genes")
    np.random.seed(random_state)
    n = X.shape[1]
    solver = _resolve_solver(svd_solver, X.shape[0], n - 1, nComp)
    if solver == "shared":
        B = _build_B_shared(X, nComp, random_state, n_oversamples)
    else:
        B = _build_B_local(X, range(n), nComp, solver, random_state, n_oversamples)
    return _assemble_A(B, n, scale, symmetric, q, as_sparse)


def _make_ray_remote():
    """Define the per-chunk ray task lazily so importing the module without ray
    installed doesn't crash."""
    @ray.remote
    def _pc_chunk_remote(X, indices, nComp, svd_solver, random_state, n_oversamples):
        return _build_B_local(X, indices, nComp, svd_solver, random_state, n_oversamples)
    return _pc_chunk_remote


def _split_indices(n, n_chunks):
    """Contiguous, balanced split of range(n) into <= n_chunks lists."""
    n_chunks = max(1, min(n_chunks, n))
    sizes = [n // n_chunks + (1 if i < n % n_chunks else 0) for i in range(n_chunks)]
    out, start = [], 0
    for s in sizes:
        if s == 0:
            continue
        out.append(list(range(start, start + s)))
        start += s
    return out


def make_pcNet(X,
               nComp: int = 3,
               scale: bool = True,
               symmetric: bool = True,
               q: float = 0.,
               as_sparse: bool = True,
               random_state: int = 0,
               n_cpus: int = 1,           # -1: use all CPUs
               timeit: bool = True,
               svd_solver: str = "auto",
               n_oversamples: int = 10):
    """Build pcNet, optionally sharding the per-gene SVDs across Ray workers."""
    if svd_solver not in _SVD_SOLVERS:
        raise ValueError(f"svd_solver={svd_solver!r}; expected one of {_SVD_SOLVERS}")
    start_time = time.time()

    X_arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
    if nComp < 2 or nComp >= X_arr.shape[1]:
        raise ValueError("nComp should be greater or equal than 2 and lower than the total number of genes")
    n = X_arr.shape[1]
    solver = _resolve_solver(svd_solver, X_arr.shape[0], n - 1, nComp)

    use_ray = n_cpus != 1 and _HAS_RAY
    if n_cpus != 1 and not _HAS_RAY:
        logger.warning("ray not installed; falling back to single-process build")

    if solver == "shared":
        # Shared-SVD path: a single sketched range-finder is reused across all
        # genes, so per-gene Ray sharding would only amortise tiny small-SVDs
        # and isn't worth the IPC. Always run locally.
        np.random.seed(random_state)
        B = _build_B_shared(X_arr, nComp, random_state, n_oversamples)
    elif use_ray:
        if n_cpus == -1:
            n_cpus = os.cpu_count() or 1
        owns_ray = not ray.is_initialized()
        if owns_ray:
            ray.init(num_cpus=n_cpus, ignore_reinit_error=True, log_to_driver=False)
            logger.info("ray init, using %d CPUs", n_cpus)

        np.random.seed(random_state)
        chunks = _split_indices(n, n_cpus)
        X_ref = ray.put(X_arr)
        pc_chunk_remote = _make_ray_remote()
        futures = [
            pc_chunk_remote.remote(X_ref, idx, nComp, solver, random_state, n_oversamples)
            for idx in chunks
        ]
        parts = ray.get(futures)
        B = np.concatenate(parts, axis=0)

        if owns_ray:
            ray.shutdown()
    else:
        np.random.seed(random_state)
        B = _build_B_local(X_arr, range(n), nComp, solver, random_state, n_oversamples)

    net = _assemble_A(B, n, scale, symmetric, q, as_sparse)
    if timeit:
        duration = time.time() - start_time
        logger.info("execution time of making pcNet: %.2f s (solver=%s)", duration, solver)
    return net


def main():
    counts = np.random.randint(0, 10, (5, 100))
    net = make_pcNet(counts, as_sparse=True, timeit=True, n_cpus=1)
    logger.info("input counts shape: %s, make pcNet completed, shape: %s", counts.shape, net.shape)
    return 100


if __name__ == "__main__":
    import sys
    sys.exit(main())
