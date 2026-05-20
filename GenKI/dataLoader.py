import numpy as np
import scipy
import anndata
import os
import hashlib
import json
import logging
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from .pcNet import make_pcNet
from .preprocessing import check_adata

logger = logging.getLogger(__name__)


def _hash_grn_inputs(matrix, params: dict) -> str:
    """Hash that uniquely identifies (matrix, build params) for GRN caching."""
    h = hashlib.sha256()
    h.update(str(matrix.shape).encode())
    h.update(str(matrix.dtype).encode())
    if scipy.sparse.issparse(matrix):
        m = matrix.tocsr()
        h.update(m.data.tobytes())
        h.update(m.indices.tobytes())
        h.update(m.indptr.tobytes())
    else:
        h.update(np.ascontiguousarray(matrix).tobytes())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()


class scBase():
    def __init__(self,
            adata: anndata.AnnData,
            target_gene: list[str],
            target_cell: str = None,
            obs_label: str = "ident",
            GRN_file_dir: str = "GRNs",
            rebuild_GRN: bool = True,
            **kwargs):

        check_adata(adata)
        self._gene_names = list(adata.var_names)
        if isinstance(target_gene[0], int): # list[int]
            target_gene = adata.var_names[target_gene].tolist()
        if all([g not in self._gene_names for g in target_gene]):
            raise IndexError("The input target gene should be in the gene list of adata")
        else:
            self._target_gene = target_gene
        if target_cell is None:
            self._counts = adata.X # use all cells, standardized counts
            logger.debug("use all the cells (%d) in adata", self._counts.shape[0])
        elif not (obs_label in adata.obs.keys()):
            raise IndexError("require a valid cell label in adata.obs")
        else:
            self._counts = adata[adata.obs[obs_label] == target_cell, :].X
        self._counts = scipy.sparse.lil_matrix(self._counts) # sparse

        cache_key = _hash_grn_inputs(adata.layers["norm"], {"nComp": 5, **kwargs})
        cache_path = os.path.join(GRN_file_dir, f"{cache_key[:12]}.npz")

        if os.path.exists(cache_path):
            logger.debug("loading GRN from cache \"%s\"", cache_path)
            self._net = scipy.sparse.load_npz(cache_path)
        elif rebuild_GRN:
            logger.debug("build GRN")
            self._net = make_pcNet(adata.layers["norm"], nComp=5, as_sparse=True, **kwargs)
            os.makedirs(GRN_file_dir, exist_ok=True)
            scipy.sparse.save_npz(cache_path, self._net)
            logger.debug("GRN built and cached at \"%s\"", cache_path)
        else:
            raise FileNotFoundError(
                f"no cached GRN at \"{cache_path}\"; pass rebuild_GRN=True to build it"
            )
        logger.debug("init completed")

    @property
    def counts(self):
        return self._counts

    @property
    def net(self):
        return self._net

    @property
    def target_gene(self):
        return self._target_gene


    def __len__(self):
        return len(self._counts)  


    def __call__(self, gene_name: list[str]): 
        return [self._gene_names.index(g) for g in gene_name]  # return gene index


    def __repr__(self):
        info = f"counts: {self._counts.shape}\n"\
        f"net: {self._net.shape}\n"\
        f"target_gene: {self._target_gene}"
        return info


class DataLoader(scBase):
    def __init__(self,
            adata: anndata.AnnData,
            target_gene: list[str],
            target_cell: str = None,
            obs_label: str = "ident",
            GRN_file_dir: str = "GRNs",
            rebuild_GRN: bool = True,
            cutoff: int = 85,
            **kwargs):
        super().__init__(adata,
                         target_gene,
                         target_cell,
                         obs_label,
                         GRN_file_dir,
                         rebuild_GRN,
                         **kwargs)
        self.cutoff = cutoff
        self.edge_index = torch.tensor(self._build_edges(), dtype = torch.long) # dense np to tensor


    def _build_edges(self, net = None):
        '''
        net: array-like, GRN built from data.
        '''
        if net is None:
            net = self._net
        grn_to_use = abs(net.toarray()) if scipy.sparse.issparse(net) else abs(net)
        grn_to_use[grn_to_use < np.percentile(grn_to_use, self.cutoff)] = 0
        edge_index_np = np.asarray(np.where(grn_to_use > 0), dtype = int)
        return edge_index_np # dense np

    def load_data(self):
        return self._data_init()

    def load_kodata(self):
        return self._KO_data_init()


    def _data_init(self):
        counts = self._counts.toarray() if scipy.sparse.issparse(self._counts) else self._counts
        x = torch.tensor(counts.T, dtype = torch.float) # define x
        return Data(x = x, edge_index = self.edge_index, y = self._gene_names)


    def _KO_data_init(self):
        # KO edges
        mask = ~(torch.isin(self.edge_index[0], torch.tensor(self(self._target_gene))) + torch.isin(self.edge_index[1], torch.tensor(self(self._target_gene))))
        edge_index_KO = self.edge_index[:, mask] # torch.long

        # KO counts
        counts_KO = self._counts.copy()
        counts_KO[:, self(self._target_gene)] = 0
        counts_KO = counts_KO.toarray() if scipy.sparse.issparse(counts_KO) else counts_KO
        x_KO = torch.tensor(counts_KO.T, dtype = torch.float) # define counts (KO)
        return Data(x = x_KO, edge_index = edge_index_KO, y = self._gene_names)


    def _gen_ZINB(self, 
                p_BIN = 0.95, 
                n_NB = 10, 
                p_NB = 0.5, 
                noise = True, 
                normalize = True, 
                show = False,
                decays: list[float] = [1.]):
        '''
        p_BIN: parameter of binomial distribution.
        n_NB, p_NB: parameters of negative binomial distribution.
        noise: add noise to ZINB samples.
        normalize: LogNorm samples, scale_factor = 10e4.
        show: show histograms of simulated samples and WT expression.
        decay: scales on n_NB simulating target gene expression.
        '''
        np.random.seed(11)   
        if decays[0] != 1.:
            raise ValueError("first element in decays should always be 1")
        elif len(decays) != len(self._target_gene):
            raise ValueError("length of decays should match target gene") 
        else: 
            s_BIN = np.random.binomial(1, p_BIN, self._counts.shape[0]) # sample binomial 0 and 1
            s = np.array([s_BIN * np.random.negative_binomial(decay*n_NB, p_NB, self._counts.shape[0]) for decay in decays]).T # ZINB
        if noise:
            s[s > 0] = s[s > 0]+ np.random.choice([-1, 0, 1], size = len(s[s > 0]), p = [0.15, 0.7, 0.15]) # add int noise
        if normalize:
            scale_factor = 10e4
            s = np.log1p(s * scale_factor / s.sum()) # LogNorm
        if show:
            fig, ax = plt.subplots(ncols = 2, figsize = (12, 5))
            counts = self._counts.toarray() if scipy.sparse.issparse(self._counts) else self._counts
            for i, (values, title) in enumerate(zip([counts[:, self(self._target_gene)], s], ['WT', 'Simulated'])):
                ax[i].hist(values, bins = counts.shape[0]//5, label = self._target_gene)
                ax[i].set_title(title)
                ax[i].legend()
            plt.show()
        logger.debug("sample gene patterns (%d) from NB%s with P(zero) = %.2f", self._counts.shape[0], (n_NB, p_NB), round(1 - p_BIN, 2))
        return s
    

    def scale_rows_and_cols(self, scale): # scale target gene edges in pcNet as a whole
        mask = np.ones(len(self._gene_names), dtype = bool)
        mask[self(self._target_gene)] = 0
        mask = np.invert(mask[:, None] @ mask[None, :])
        self._net[mask] *= scale 


    def OE_data_init(self, 
            weight_scale: list[float] = [10.,], 
            gene_patterns: list[float] = None, 
            **kwargs):
        # OE edges
        weight_scale = np.asarray(weight_scale)
        net_OE = self._net.toarray().copy()
        net_OE[:, self(self._target_gene)] *= weight_scale[None, :]
        net_OE[self(self._target_gene), :] *= weight_scale[:, None] # note cross points get multiplied twice

        edge_index_OE = self._build_edges(net_OE)
        edge_index_OE = torch.tensor(edge_index_OE, dtype = torch.long)

        # OE counts
        counts_OE = self._counts.copy()
        orig_counts = counts_OE[:, self(self._target_gene)]
        if gene_patterns is not None:
            counts_OE[:, self(self._target_gene)] = gene_patterns
        else:
            decays = [w/weight_scale[0] for w in weight_scale] # n_NB = coeff * n_NB(10), coeff <= 1
            counts_OE[:, self(self._target_gene)] = self._gen_ZINB(n_NB = weight_scale[0], decays = decays, **kwargs) 
        counts_OE = counts_OE.toarray() if scipy.sparse.issparse(counts_OE) else counts_OE
        x_OE = torch.tensor(counts_OE.T, dtype = torch.float) 
        logger.debug("replace expression of %s to simulated expressions and edges by scale %s", self._target_gene, weight_scale)
        return Data(x = x_OE, edge_index = edge_index_OE, y = self._gene_names)
