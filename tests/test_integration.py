"""End-to-end pin: build_adata -> DataLoader -> VGAE_trainer -> ranking.

Exercises the documented workflow on the bundled sample dataset. Skipped
automatically if torch / torch_geometric are unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from GenKI.dataLoader import DataLoader
from GenKI.preprocessing import build_adata
from GenKI.train import VGAE_trainer
from GenKI.utils import get_distance, get_generank


def test_full_workflow(small_adata, tmp_path):
    adata = build_adata(small_adata, log_normalize=False, scale_data=True)
    assert "norm" in adata.layers
    assert (adata.X.toarray() < 0).any()  # standardized

    target_gene = [adata.var_names[0]]
    loader = DataLoader(
        adata,
        target_gene=target_gene,
        GRN_file_dir=str(tmp_path / "GRNs"),
    )
    n_genes = adata.n_vars
    assert loader.net.shape == (n_genes, n_genes)

    data = loader.load_data()
    data_ko = loader.load_kodata()
    assert data.x.shape == (n_genes, adata.n_obs)
    assert data_ko.x.shape == (n_genes, adata.n_obs)

    sensei = VGAE_trainer(data, epochs=3, seed=0, verbose=False)
    sensei.train()
    step, loss, auc, ap = sensei.final_metrics
    assert step == 3
    assert np.isfinite([loss, auc, ap]).all()

    z_m, z_S = sensei.get_latent_vars(data)
    z_m_ko, z_S_ko = sensei.get_latent_vars(data_ko)
    assert z_m.shape == (n_genes, 2)

    dis = get_distance(z_m_ko, z_S_ko, z_m, z_S, by="KL")
    assert dis.shape == (n_genes,)
    assert np.isfinite(dis).all()

    df = get_generank(data, dis)
    assert len(df) == n_genes
    assert "rank" in df.columns
    assert list(df["rank"]) == list(range(1, n_genes + 1))


def test_grn_cache_roundtrip(small_adata, tmp_path):
    """Building twice with the same inputs hits the on-disk cache the second
    time and produces an identical GRN — no rebuild needed."""
    import time
    import os
    adata = build_adata(small_adata, log_normalize=False, scale_data=True)
    target_gene = [adata.var_names[0]]
    grn_dir = str(tmp_path / "GRNs")

    t0 = time.time()
    first = DataLoader(
        adata, target_gene=target_gene, GRN_file_dir=grn_dir,
    )
    t_first = time.time() - t0

    # Exactly one cache file written, named with the content hash.
    cached = [f for f in os.listdir(grn_dir) if f.endswith(".npz")]
    assert len(cached) == 1, f"expected exactly one cache file, found {cached}"

    t0 = time.time()
    second = DataLoader(
        adata, target_gene=target_gene, rebuild_GRN=False,
        GRN_file_dir=grn_dir,
    )
    t_second = time.time() - t0

    assert (first.net != second.net).nnz == 0, "cached GRN must match the freshly built one"
    # A real cache hit should be at least ~5x faster than the original build.
    assert t_second < t_first / 2, f"cache hit not faster: {t_first:.3f}s vs {t_second:.3f}s"
