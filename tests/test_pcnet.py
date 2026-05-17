"""Behavioural pins for GenKI.pcNet (no torch required)."""

import numpy as np
import pytest

from GenKI.pcNet import make_pcNet, pcNet


def _counts(seed=0, cells=30, genes=8):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=(cells, genes)).astype(float)


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
