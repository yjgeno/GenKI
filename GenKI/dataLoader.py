import numpy as np
import scipy
import anndata
import os
import torch 
from torch_geometric.data import Data 
import matplotlib.pyplot as plt
# from tqdm import tqdm
from .pcNet import make_pcNet
from .preprocesing import check_adata


class scBase():
    def __init__(self, 
            adata: anndata.AnnData, 
            target_gene: list[str], 
            target_cell: str = None, 
            obs_label: str = "ident",
            GRN_file_dir: str = "GRNs",
            rebuild_GRN: bool = False, 
            pcNet_name: str = "pcNet", 
            verbose: bool = False,
            **kwargs): 

        check_adata(adata)
        self._gene_names = list(adata.var_names)
        if all([g not in self._gene_names for g in target_gene]):
            raise IndexError("The input target gene should be in the gene list of adata")
        else:
            self._target_gene = target_gene
            # self._idx_KO = self._gene_names.index(target_gene)
        if target_cell is None:
            self._counts = adata.X # use all cells, standardized counts
            if verbose:
                print(f"use all the cells ({self._counts.shape[0]}) in adata")
        elif not (obs_label in adata.obs.keys()):
            raise IndexError("require a valid cell label in adata.obs")
        else:
            self._counts = adata[adata.obs[obs_label] == target_cell, :].X 
        self._counts = scipy.sparse.lil_matrix(self._counts) # sparse 
        pcNet_path = os.path.join(GRN_file_dir, f"{pcNet_name}.npz")
        if rebuild_GRN: 
            if verbose:
                print("build GRN")
            self._net = make_pcNet(adata.layers["norm"], nComp = 5, as_sparse = True, timeit = verbose, **kwargs)           
            os.makedirs(GRN_file_dir, exist_ok = True) # create dir 
            scipy.sparse.save_npz(pcNet_path, self._net) # save GRN
            # scipy.sparse.lil_matrix(pcNet_np) # np to sparse
            if verbose:
                print(f"GRN has been built and saved in \"{pcNet_path}\"")
        else:
            try:
                if verbose:
                    print(f"loading GRN from \"{pcNet_path}\"")
                self._net = scipy.sparse.load_npz(pcNet_path)
            except ImportError:
                print("require npz file name")
        if verbose:
            print("init completed\n")  

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
            rebuild_GRN: bool = False, 
            pcNet_name: str = "pcNet", 
            cutoff: int = 85,
            verbose: bool = True,
            **kwargs):
        super().__init__(adata, 
                         target_gene, 
                         target_cell, 
                         obs_label, 
                         GRN_file_dir, 
                         rebuild_GRN, 
                         pcNet_name, 
                         verbose, 
                         **kwargs)
        self.verbose = verbose
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
        if self.verbose:
            print(f"set expression of {self._target_gene} to zeros and remove edges")
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
        if self.verbose:
            print(f"sample gene patterns ({self._counts.shape[0]}) from NB{n_NB, p_NB} with P(zero) = {round(1-p_BIN, 2)}")
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
        if self.verbose:
            print(f"replace expression of {self._target_gene} to simulated expressions and edges by scale {weight_scale}")
        return Data(x = x_OE, edge_index = edge_index_OE, y = self._gene_names)


    # def run_sys_KO(self, model, genelist):
    #     '''
    #     model: a trained VGAE model.
    #     genelist: array-like, gene list to be systematic KO
    #     '''
    #     self.verbose = False
    #     g_orig = self._target_gene 
    #     data = self.data_init()
    #     z_m0, z_S0 = get_latent_vars(data, model)
    #     sys_res = []
    #     from tqdm import tqdm
    #     for g in tqdm(genelist, desc = "systematic KO...", total = len(genelist)):
    #         if g not in self._gene_names:
    #             raise IndexError(f'"{g}" is not in the object')
    #         else:  
    #             self._target_gene = g # reset KO gene
    #             data_v = self.KO_data_init()
    #             z_mv, z_Sv = get_latent_vars(data_v, model)
    #             dis = get_distance(z_mv, z_Sv, z_m0, z_S0, by = "KL")
    #             sys_res.append(dis)
    #     self._target_gene = g_orig
    #     # print(self._target_gene)
    #     self.verbose = True
    #     return np.array(sys_res) 
