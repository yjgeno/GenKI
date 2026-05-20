"""High-level, one-call interface for the GenKI workflow.

Example
-------
>>> from GenKI import GenKI
>>> ranked = GenKI.from_h5ad("data/microglial_seurat_WT.h5ad", ["GNB4"]).run(
...     epochs=100, seed=0, n_permutations=100
... )
>>> ranked.head()

The facade wires together the four lower-level pieces (``build_adata`` ->
``DataLoader`` -> ``VGAE_trainer`` -> ``get_distance``/``get_generank``).
Power users can still reach the underlying objects via ``.loader`` and
``.trainer``.
"""

from __future__ import annotations

from .dataLoader import DataLoader
from .preprocessing import build_adata
from .train import VGAE_trainer
from .utils import get_distance, get_generank


class GenKI:
    def __init__(
        self,
        adata,
        target_gene,
        *,
        target_cell=None,
        obs_label: str = "ident",
        cutoff: int = 85,
        grn_dir: str = "GRNs",
        rebuild_grn: bool = True,
        **grn_kwargs,
    ):
        self.target_gene = list(target_gene)
        self._loader = DataLoader(
            adata,
            target_gene=self.target_gene,
            target_cell=target_cell,
            obs_label=obs_label,
            GRN_file_dir=grn_dir,
            rebuild_GRN=rebuild_grn,
            cutoff=cutoff,
            **grn_kwargs,
        )
        self.wt_data = self._loader.load_data()
        self.ko_data = self._loader.load_kodata()
        self._trainer = None
        self.metrics = None

    @classmethod
    def from_h5ad(
        cls,
        path: str,
        target_gene,
        *,
        log_normalize: bool = False,
        scale_data: bool = True,
        as_sparse: bool = True,
        uppercase: bool = True,
        build_kwargs: dict | None = None,
        **kwargs,
    ) -> "GenKI":
        """Read an ``.h5ad`` file, preprocess it, and build the workflow.

        Note: with ``uppercase=True`` (default) gene names are upper-cased,
        so ``target_gene`` should be given in upper case.
        """
        adata = build_adata(
            path,
            log_normalize=log_normalize,
            scale_data=scale_data,
            as_sparse=as_sparse,
            uppercase=uppercase,
            **(build_kwargs or {}),
        )
        return cls(adata, target_gene, **kwargs)

    @classmethod
    def from_adata(
        cls,
        adata,
        target_gene,
        *,
        preprocess: bool = False,
        log_normalize: bool = False,
        scale_data: bool = True,
        as_sparse: bool = True,
        uppercase: bool = True,
        build_kwargs: dict | None = None,
        **kwargs,
    ) -> "GenKI":
        """Build the workflow from an in-memory AnnData.

        Set ``preprocess=True`` to run ``build_adata`` first; otherwise the
        AnnData is assumed to already be normalized/standardized with a
        ``layers["norm"]``.
        """
        if preprocess:
            adata = build_adata(
                adata,
                log_normalize=log_normalize,
                scale_data=scale_data,
                as_sparse=as_sparse,
                uppercase=uppercase,
                **(build_kwargs or {}),
            )
        return cls(adata, target_gene, **kwargs)

    def fit(
        self,
        *,
        epochs: int = 100,
        lr: float = 7e-4,
        weight_decay: float = 9e-4,
        beta: float = 1e-4,
        out_channels: int = 2,
        seed: int | None = None,
        log_dir: str | None = None,
        verbose: bool = False,
        **train_kwargs,
    ) -> "GenKI":
        self._trainer = VGAE_trainer(
            self.wt_data,
            out_channels=out_channels,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            beta=beta,
            log_dir=log_dir,
            verbose=verbose,
            seed=seed,
        )
        self._trainer.train(**train_kwargs)
        self.metrics = self._trainer.final_metrics
        return self

    def predict(self, *, n_permutations: int = 100, by: str = "KL", **generank_kwargs):
        """Rank genes perturbed by the knock-out. Returns a pandas DataFrame."""
        if self._trainer is None:
            raise RuntimeError("call fit() (or run()) before predict()")
        z_m, z_S = self._trainer.get_latent_vars(self.wt_data)
        z_m_ko, z_S_ko = self._trainer.get_latent_vars(self.ko_data)
        dis = get_distance(z_m_ko, z_S_ko, z_m, z_S, by=by)
        null = (
            self._trainer.pmt(self.ko_data, n=n_permutations, by=by)
            if n_permutations
            else None
        )
        return get_generank(self.wt_data, dis, null, **generank_kwargs)

    def run(
        self,
        *,
        epochs: int = 100,
        lr: float = 7e-4,
        weight_decay: float = 9e-4,
        beta: float = 1e-4,
        out_channels: int = 2,
        seed: int | None = None,
        log_dir: str | None = None,
        verbose: bool = False,
        n_permutations: int = 100,
        by: str = "KL",
        generank_kwargs: dict | None = None,
        **train_kwargs,
    ):
        """Convenience: ``fit`` then ``predict`` in a single call."""
        self.fit(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            beta=beta,
            out_channels=out_channels,
            seed=seed,
            log_dir=log_dir,
            verbose=verbose,
            **train_kwargs,
        )
        return self.predict(
            n_permutations=n_permutations, by=by, **(generank_kwargs or {})
        )

    @property
    def loader(self) -> DataLoader:
        return self._loader

    @property
    def trainer(self):
        return self._trainer

    def __repr__(self) -> str:
        n_genes, n_cells = self.wt_data.x.shape
        return (
            f"GenKI(target_gene={self.target_gene}, genes={n_genes}, "
            f"cells={n_cells}, fitted={self._trainer is not None})"
        )
