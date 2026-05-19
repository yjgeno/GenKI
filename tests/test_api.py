"""End-to-end pin for the high-level GenKI facade."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from GenKI import GenKI


def test_facade_run_returns_ranked_dataframe(small_adata, tmp_path):
    adata_target = str(small_adata.var_names[0]).upper()  # build_adata upper-cases

    gk = GenKI.from_adata(
        small_adata,
        [adata_target],
        preprocess=True,
        rebuild_grn=True,
        grn_dir=str(tmp_path / "GRNs"),
    )
    assert "fitted=False" in repr(gk)

    n_genes = gk.wt_data.x.shape[0]
    ranked = gk.run(epochs=3, seed=0, n_permutations=0)

    assert gk.trainer is not None
    step, loss, auc, ap = gk.metrics
    assert step == 3
    assert np.isfinite([loss, auc, ap]).all()

    assert "rank" in ranked.columns
    assert len(ranked) == n_genes
    assert ranked["rank"].tolist() == list(range(1, n_genes + 1))


def test_predict_before_fit_raises(small_adata, tmp_path):
    gk = GenKI.from_adata(
        small_adata,
        [str(small_adata.var_names[0]).upper()],
        preprocess=True,
        rebuild_grn=True,
        grn_dir=str(tmp_path / "GRNs"),
    )
    with pytest.raises(RuntimeError):
        gk.predict(n_permutations=0)
