from pathlib import Path
from anndata import (
    AnnData,
    read_h5ad,
    read_csv,
    read_excel,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
)
import scanpy as sc
import pandas as pd
from typing import Union
from scipy import sparse

sc.settings.verbosity = 0

# https://scanpy.readthedocs.io/en/stable/api.html#reading
file_attrs = ["h5ad", "csv", "xlsx", "h5", "loom", "mtx", "txt", "mat"]
read_fun_keys = ["h5ad", "csv", "excel", "hdf", "loom", "mtx", "text"] # fun from scanpy read


def _read_counts(counts_path: str, 
                transpose: bool = False, 
                **kwargs):

    """Read counts file to build an AnnData.
    Args:
        counts_path (str): Path to counts file.
        transpose (bool, optional): Whether transpose the counts.
    Returns:
        AnnData
    """

    file_attr = counts_path.split(".")[-1]
    if Path(counts_path).is_file() and file_attr in file_attrs:
        print(f"load counts from {counts_path}")
        if file_attr == "mat":
            import numpy as np
            import h5py
            f = h5py.File(counts_path,'r')
            # print(f.keys())
            counts = np.array(f.get(list(f.keys())[0]), dtype='float32')
            if transpose:
                counts = counts.T
            adata = sc.AnnData(counts)
        else:
            read_fun_key = read_fun_keys[file_attrs.index(file_attr)]
            read_fun = getattr(sc, f"read_{read_fun_key}")  # define sc.read_{} function
            if transpose:
                adata = read_fun(counts_path, **kwargs).transpose() # transpose counts file
            else:
                adata = read_fun(counts_path, **kwargs)
    else:
        raise ValueError("incorrect file path given to counts")
    return adata


def build_adata(
    counts_path: str,
    meta_gene_path: Union[None, str] = None,
    meta_cell_path: Union[None, str] = None,
    meta_cell_cols: Union[None, str] = None,
    sep = "\t",
    header = None,
    log_normalize: bool = False,
    scale_data: bool = False,
    as_sparse: bool = True,
    **kwargs,
):

    """Load counts, metadata of genes and cells to build an AnnData input for scTenifoldXct.
    Args:
        counts_path (str): Path to counts.
        meta_gene (Union[None, str]): Path to metadata of variables.
        meta_cell (Union[None, str]): Path to metadata of Observations
        sep (str, optional): The delimiter for metadata. Defaults to "\t".
        log_normalize (bool, optional): Whether log-normalize the counts. Defaults to False.
        scale_data (bool, optional): Whether standardize the counts. Defaults to False.
        as_sparse (bool, optional): Whether make the counts sparse. Defaults to True.
        **kwargs: key words in _read_counts
    Returns:
        AnnData
    """
    
    adata = _read_counts(counts_path, **kwargs)
   
    if meta_gene_path is not None and Path(meta_gene_path).is_file():
        try:
            print("add metadata for genes")
            adata.var_names = pd.read_csv(meta_gene_path, header=header, sep=sep)[0]
        except Exception:
            raise ValueError("incorrect file path given to meta_gene")
    if meta_cell_path is not None and Path(meta_cell_path).is_file():
        try:
            print("add metadata for cells")
            adata.obs = pd.read_csv(meta_cell_path, header=header, sep=sep)
            if meta_cell_cols is not None:
                adata.obs.columns = meta_cell_cols
        except Exception:
            raise ValueError("incorrect file path given to meta_cell")

    if log_normalize:
        print("normalize counts")
        adata.layers["raw"] = adata.X
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)


    if scale_data:
        adata.layers["norm"] = adata.X
        print("standardize counts")
        from sklearn import preprocessing
        counts = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        scaler = preprocessing.StandardScaler().fit(counts)
        adata.X = scaler.transform(counts)

    if as_sparse:
        print("make counts sparse")
        adata.X = (
            sparse.csr_matrix(adata.X) if not sparse.issparse(adata.X) else adata.X
        )
    adata.var_names = adata.var_names.str.upper() # all species use upper case genes

    return adata


def check_adata(adata):
    if "norm" not in adata.layers:
        raise ValueError("normalized counts should be saved at adata.layers[\"norm\"]")
    if not (adata.X.toarray() < 0).any(): 
        raise ValueError("adata.X should be standardized counts")


if __name__ == "__main__":
    from os import path

    data_dir = path.join(
        path.dirname(path.dirname(path.abspath(__file__))), "filtered_gene_bc_matrices"
    )
    counts_path = str(path.join(data_dir, "matrix.mtx"))
    gene_path = str(path.join(data_dir, "genes.tsv"))
    cell_path = str(path.join(data_dir, "barcodes.tsv"))
    adata = build_adata(counts_path, gene_path, cell_path)
    print(adata)
