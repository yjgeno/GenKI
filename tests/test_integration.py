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
        rebuild_GRN=True,
        GRN_file_dir=str(tmp_path / "GRNs"),
        pcNet_name="pcNet",
        verbose=False,
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
