"""Behavioural pins for GenKI.pcNet (no torch required)."""

import numpy as np
import pytest

from GenKI.pcNet import make_pcNet, pcNet


def _counts(seed=0, cells=30, genes=8):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=(cells, genes)).astype(float)


def _lowrank(seed=0, cells=200, genes=60, rank=8, noise=0.05):
    """A low-rank+noise matrix; matches the spectral structure of real scRNA-seq
    well enough for downstream parity checks on the top-edge set."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((cells, rank))
    V = rng.standard_normal((rank, genes))
    return U @ V + noise * rng.standard_normal((cells, genes))


def test_pcnet_shape_symmetry_and_zero_diagonal():
    X = _counts()
    A = pcNet(X, nComp=3, symmetric=True, as_sparse=False, random_state=0)
    assert A.shape == (X.shape[1], X.shape[1])
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0.0)


def test_pcnet_is_deterministic():
    X = _counts()
    a = pcNet(X, nComp=3, as_sparse=False, random_state=0)
    b = pcNet(X, nComp=3, as_sparse=False, random_state=0)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bad_ncomp", [1, 8, 20])
def test_pcnet_ncomp_validation(bad_ncomp):
    X = _counts()  # 8 genes
    with pytest.raises(ValueError):
        pcNet(X, nComp=bad_ncomp, as_sparse=False)


def test_make_pcnet_single_cpu_matches_pcnet():
    X = _counts()
    direct = pcNet(X, nComp=3, as_sparse=False, random_state=0)
    via_make = make_pcNet(
        X, nComp=3, as_sparse=False, random_state=0, n_cpus=1, timeit=False
    )
    assert np.allclose(direct, via_make)


def test_invalid_svd_solver_raises():
    X = _counts()
    with pytest.raises(ValueError, match="svd_solver"):
        pcNet(X, nComp=3, as_sparse=False, svd_solver="nope")


def test_auto_solver_picks_full_for_small_inputs():
    """On a small problem `auto` must be bit-identical to `full` (legacy)."""
    X = _counts()
    full = pcNet(X, nComp=3, as_sparse=False, svd_solver="full", random_state=0)
    auto = pcNet(X, nComp=3, as_sparse=False, svd_solver="auto", random_state=0)
    assert np.array_equal(full, auto)


@pytest.mark.parametrize("solver", ["full", "randomized", "lanczos"])
def test_solver_parity_on_structured_data(solver):
    """All non-shared solvers agree on structured data to working precision."""
    X = _lowrank()
    full = pcNet(X, nComp=5, as_sparse=False, svd_solver="full", random_state=0)
    other = pcNet(X, nComp=5, as_sparse=False, svd_solver=solver, random_state=0)
    # randomized/lanczos use truncated solvers; for structured data the
    # absolute error from singular-subspace rotation is ~1e-10.
    assert np.allclose(full, other, atol=1e-8, rtol=1e-6), (
        f"{solver} max|diff|={np.max(np.abs(full - other)):.2e}"
    )


@pytest.mark.parametrize("solver", ["randomized", "lanczos", "shared"])
def test_solver_determinism(solver):
    """Truncated solvers must give bit-identical output for the same seed."""
    X = _lowrank()
    a = pcNet(X, nComp=5, as_sparse=False, svd_solver=solver, random_state=7)
    b = pcNet(X, nComp=5, as_sparse=False, svd_solver=solver, random_state=7)
    assert np.array_equal(a, b)


def test_shared_solver_preserves_top_edges_on_structured_data():
    """Shared-SVD is an approximation; pin its quality on a representative
    low-rank+noise matrix so future regressions show up here."""
    X = _lowrank()
    full = pcNet(X, nComp=5, as_sparse=False, svd_solver="full", random_state=0)
    shared = pcNet(X, nComp=5, as_sparse=False, svd_solver="shared", random_state=0)

    def top_edge_mask(A, q=85):
        return np.abs(A) >= np.percentile(np.abs(A), q)

    ref = top_edge_mask(full)
    got = top_edge_mask(shared)
    jaccard = (ref & got).sum() / (ref | got).sum()
    assert jaccard >= 0.95, f"shared top-edge Jaccard = {jaccard:.3f}"


@pytest.mark.parametrize("n_cpus", [1, 2])
def test_make_pcnet_n_cpus_bit_identical_full_solver(n_cpus):
    """Ray sharding with the full solver must reproduce single-CPU output."""
    X = _lowrank(cells=100, genes=40, rank=6)
    a = make_pcNet(
        X, nComp=3, as_sparse=False, random_state=0, n_cpus=1, timeit=False, svd_solver="full"
    )
    b = make_pcNet(
        X, nComp=3, as_sparse=False, random_state=0, n_cpus=n_cpus, timeit=False, svd_solver="full"
    )
    assert np.array_equal(a, b)
