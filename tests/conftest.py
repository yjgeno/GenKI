import warnings
from pathlib import Path

import numpy as np
import pytest

# anndata>=0.10 emits FutureWarnings for the top-level read_* names that
# GenKI.preprocesing still imports. They are non-fatal; silence so the
# behavioural assertions are not masked by warning noise.
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "microglial_seurat_WT.h5ad"


@pytest.fixture(scope="session")
def sample_h5ad_path() -> str:
    if not DATA_FILE.is_file():
        pytest.skip(f"sample dataset not available: {DATA_FILE}")
    return str(DATA_FILE)


@pytest.fixture(scope="session")
def small_adata(sample_h5ad_path):
    """A deterministically down-sampled raw AnnData for fast integration tests."""
    import anndata as ad

    adata = ad.read_h5ad(sample_h5ad_path)
    rng = np.random.default_rng(0)
    n_cells = min(300, adata.n_obs)
    n_genes = min(50, adata.n_vars)
    cell_idx = np.sort(rng.choice(adata.n_obs, size=n_cells, replace=False))
    gene_idx = np.arange(n_genes)
    return adata[cell_idx, gene_idx].copy()
