# GenKI — Gene Knock-out Inference

[![PyPI version](https://img.shields.io/pypi/v/genki.svg)](https://pypi.org/project/genki/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fnar%2Fgkad450-blue)](https://doi.org/10.1093/nar/gkad450)

A Variational Graph Auto-Encoder (VGAE) model for predicting gene perturbation effects from scRNA-seq data. GenKI performs *in silico* gene knock-out experiments on a gene regulatory network (GRN) without requiring real knock-out data.

<p align="center">
    <img src="logo.jpg" alt="GenKI logo" width="300"/>
</p>

## Prerequisites

GenKI requires **PyTorch** and **PyTorch Geometric**, which must be installed separately to match your CUDA version:

1. [Install PyTorch](https://pytorch.org/get-started/locally/)
2. [Install PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Installation

```shell
pip install GenKI
```

Or install directly from source:

```shell
pip install git+https://github.com/yjgeno/GenKI.git
```

Or with conda (sets up the full environment):

```shell
conda env create -f environment.yml
conda activate ogenki
```

## Quick Start

```python
import scanpy as sc
from GenKI.preprocesing import build_adata
from GenKI.dataLoader import DataLoader
from GenKI.train import VGAE_trainer
from GenKI import utils

# 1. Load and preprocess data
adata = build_adata("data/my_data.h5ad")

# 2. Build GRN and prepare WT / virtual-KO graph data
data_wrapper = DataLoader(
    adata,
    target_gene=["TUBG1"],   # gene to knock out
    target_cell=None,         # None = use all cells
    GRN_file_dir="GRNs",
    rebuild_GRN=True,
    pcNet_name="pcNet",
    verbose=True,
    n_cpus=8,
)
data_wt = data_wrapper.load_data()
data_ko = data_wrapper.load_kodata()

# 3. Train VGAE
sensei = VGAE_trainer(data_wt, epochs=100, lr=7e-4, beta=1e-4, seed=8096)
sensei.train()

# 4. Get latent distributions and compute KL divergence per gene
z_mu_wt, z_std_wt = sensei.get_latent_vars(data_wt)
z_mu_ko, z_std_ko = sensei.get_latent_vars(data_ko)
dis = utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")

# 5. Rank genes by perturbation effect (with permutation test)
null = sensei.pmt(data_ko, n=100, by="KL")
res = utils.get_generank(data_wt, dis, null)
print(res)
```

## API

| Symbol | Description |
|---|---|
| `GenKI.dataLoader.DataLoader` | Wraps an `AnnData` object, builds/loads the GRN, and produces PyG `Data` objects for WT and virtual-KO conditions |
| `GenKI.train.VGAE_trainer` | Trains the VGAE, exposes latent variables, permutation testing, and model save/load |
| `GenKI.utils.get_distance` | Computes per-gene distribution distance (KL, EMD, t-test) between two latent spaces |
| `GenKI.utils.get_generank` | Ranks genes by perturbation score; optionally filters by permutation-test significance |
| `GenKI.preprocesing.build_adata` | Loads an `.h5ad` file and adds a log-normalised layer used by `DataLoader` |
| `GenKI.pcNet.make_pcNet` | Builds a principal-component-based GRN from expression data (parallelised with Ray) |

## Tutorial

Step-by-step virtual KO example:
[notebook/Example.ipynb](https://github.com/yjgeno/GenKI/blob/master/notebook/Example.ipynb)

## Citation

If you use GenKI in your research, please cite:

> Yang Y, Wang M, Ni P, Zhong J. *GenKI: Virtual gene knockout inference with variational graph autoencoder*. Nucleic Acids Research, 2023. https://doi.org/10.1093/nar/gkad450
