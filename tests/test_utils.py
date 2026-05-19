"""Behavioural pins for GenKI.utils numeric helpers (no torch required)."""

import math
import types

import numpy as np
import pytest

from GenKI.utils import (
    _kl_1d,
    _kl_mvn,
    _t_stat,
    _wasserstein_dist2,
    get_distance,
    get_generank,
    get_r2_score,
)


def test_kl_mvn_identical_is_zero():
    m = np.array([0.0, 0.0])
    S = np.eye(2)
    assert _kl_mvn(m, S, m, S) == pytest.approx(0.0, abs=1e-9)


def test_kl_1d_identical_is_zero():
    assert _kl_1d(1.5, 2.0, 1.5, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_kl_1d_known_value():
    # 0.5*log(1) + (1 + (0-1)^2)/(2*1) - 0.5 == 0.5
    assert _kl_1d(0.0, 1.0, 1.0, 1.0) == pytest.approx(0.5, abs=1e-12)


def test_t_stat_known_value():
    assert _t_stat(3.0, 1.0, 1.0, 1.0) == pytest.approx(math.sqrt(2.0), rel=1e-9)


def test_wasserstein_identical_is_zero():
    m = np.array([0.0, 0.0])
    S = np.eye(2)
    assert _wasserstein_dist2(m, S, m, S) == pytest.approx(0.0, abs=1e-9)


def test_get_distance_kl_2d_identical_is_zero():
    z_m = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    z_S = np.full((4, 2), 0.5)
    dis = get_distance(z_m, z_S, z_m, z_S, by="KL")
    assert dis.shape == (4,)
    assert np.allclose(dis, 0.0, atol=1e-9)


def test_get_distance_1d_t_path():
    z_m0 = np.array([3.0, 0.0])
    z_S0 = np.array([1.0, 1.0])
    z_m1 = np.array([1.0, 0.0])
    z_S1 = np.array([1.0, 1.0])
    dis = get_distance(z_m0, z_S0, z_m1, z_S1, by="t")
    assert dis[0] == pytest.approx(math.sqrt(2.0), rel=1e-9)
    assert dis[1] == pytest.approx(0.0, abs=1e-12)


def test_get_generank_ranks_by_descending_distance_without_null():
    # Current behaviour: largest distance is ranked first (most affected gene).
    data = types.SimpleNamespace(y=["a", "b", "c"])
    distance = np.array([3.0, 1.0, 2.0])
    df = get_generank(data, distance, null=None)
    assert list(df.index) == ["a", "c", "b"]
    assert list(df["rank"]) == [1, 2, 3]
    assert list(df["dis"]) == [3.0, 2.0, 1.0]


def test_get_r2_score_perfect_collinear():
    import torch

    rng = np.random.default_rng(1)
    cells = 80
    g0 = rng.normal(size=cells)
    g1 = rng.normal(size=cells)
    g2 = 2.0 * g0  # perfectly explained by g0
    x = torch.tensor(np.vstack([g0, g1, g2]), dtype=torch.float)  # genes x cells
    data = types.SimpleNamespace(x=x, y=["g0", "g1", "g2"])
    r2, r2_adj = get_r2_score(data, ["g0"], "g2")
    assert r2 == pytest.approx(1.0, abs=1e-4)
    assert r2_adj == pytest.approx(1.0, abs=1e-4)
