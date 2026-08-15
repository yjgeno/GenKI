# GenKI — Gene Knock-out Inference

[![PyPI version](https://img.shields.io/pypi/v/GenKI.svg)](https://pypi.org/project/GenKI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fnar%2Fgkad450-blue)](https://doi.org/10.1093/nar/gkad450)

A Variational Graph Auto-Encoder (VGAE) model for predicting gene perturbation effects from scRNA-seq data. GenKI performs *in silico* gene knock-out experiments on a gene regulatory network (GRN) without requiring real knock-out data.

<p align="center">
    <img src="logo.jpg" alt="GenKI logo" width="300"/>
</p>

> 🆕 **New: a local web UI.** Run GenKI from your browser — no code required.
> Requires **Python ≥ 3.10**.
> ```shell
> pip install --no-cache-dir "GenKI[web]"
> genki-ui
> ```
> Upload a `.h5ad`, pick a gene to knock out, and get ranked results in a few
> clicks. See [Web UI](#web-ui) below.

<p align="center">
    <img src="docs/webapp-screenshot.png" alt="GenKI local web UI: load a dataset, pick target genes, and configure a knock-out run" width="600"/>
</p>

## Prerequisites

**Python ≥ 3.10.** PyTorch and PyTorch Geometric install automatically (CPU builds). For GPU/CUDA, install them first to match your CUDA version: [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Installation

```shell
pip install GenKI
```

Or with conda (sets up the full environment):

```shell
conda env create -f environment.yml
conda activate ogenki
```

## Quick Start

A real microglial (wild-type) scRNA-seq dataset is bundled at
[`data/microglial_seurat_WT.h5ad`](data/microglial_seurat_WT.h5ad) so you can
run GenKI immediately. The `GenKI` facade runs the whole workflow — load &
preprocess, build the GRN, train the VGAE, rank genes — in one call:

```python
from GenKI import GenKI

ranked = GenKI.from_h5ad(
    "data/microglial_seurat_WT.h5ad",
    target_gene=["TUBG1"],   # gene(s) to knock out (upper-cased by default)
).run(epochs=100, seed=8096, n_permutations=100)

print(ranked)   # genes ranked by perturbation effect
```

Use `.fit()`/`.predict()` separately to inspect the model in between, or
`GenKI.from_adata(adata, target_gene=[...], preprocess=True)` to start from
an in-memory `AnnData`. For fine-grained control over each step (data
loading, GRN construction, training, ranking), see the collapsed example
below. Building the GRN in parallel needs the optional Ray extra
(`pip install "GenKI[ray]"`); pass `n_cpus` as a keyword argument, e.g.
`GenKI.from_h5ad(..., rebuild_grn=True, n_cpus=8)`.

## Web UI

Prefer clicking over scripting? Install the `web` extra and launch a local
UI — upload a `.h5ad`, pick a target gene, and run the workflow from the
browser, no code required. Requires **Python ≥ 3.10**; on an older
interpreter pip silently installs an ancient release without the `web`
extra:

```shell
pip install --no-cache-dir "GenKI[web]"
genki-ui
# opens http://127.0.0.1:8931 (pass --port to use a different one)
```

From a git checkout you can also skip the upload step and use the bundled
example dataset directly (it isn't shipped in the PyPI package). Everything
runs locally; no data leaves your machine.

Pick one or more genes to knock out, tune epochs/learning rate/permutations
if you like, and hit **Run**. Hover the ⓘ next to a field for what it does.
The GRN build defaults to all local CPUs (**Parallel workers** = `-1`, via
the bundled Ray extra); set a positive integer to cap it, or `1` for a
single process (`0` is invalid).

## GRN build time

`make_pcNet` fits a leave-one-out principal-component regression for every gene, so cost grows roughly as **O(genes² × cells × nComp)**. Wall-clock seconds on an Apple M1 Pro (8 cores, 16 GB RAM) under the default settings (`nComp=3`, `svd_solver="auto"`, `n_cpus=8`):

| cells \ genes | 1 000 | 3 000 | 5 000 |
|---:|---:|---:|---:|
| **500**  | 11 s | 29 s | 77 s |
| **1 000** | 12 s | 52 s | 2 min 13 s |
| **2 000** | 18 s | 1 min 37 s | 4 min 22 s |

For reference, `notebook/Example.ipynb` (1 139 cells × 3 000 genes, `n_cpus=8`) builds the GRN in about **1 minute**. Cost scales roughly linearly in `cells` and quadratically in `genes`; `n_cpus > 1` needs the optional Ray extra and pays a fixed ~10 s startup cost. GRNs are cached under `GRN_file_dir` (default `GRNs/`) — pass `rebuild_grn=True` only when `cells`/`genes`/`nComp` change.

<details>
<summary><b>Lower-level API</b> (fine-grained control over each step)</summary>

```python
from GenKI.preprocessing import build_adata
from GenKI.dataLoader import DataLoader
from GenKI.train import VGAE_trainer
from GenKI import utils

# 1. Load and preprocess data
adata = build_adata("data/microglial_seurat_WT.h5ad")

# 2. Build GRN and prepare WT / virtual-KO graph data
data_wrapper = DataLoader(
    adata,
    target_gene=["TUBG1"],   # gene to knock out
    target_cell=None,         # None = use all cells
    GRN_file_dir="GRNs",
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

</details>

## Tutorial

Step-by-step virtual KO example:
[notebook/Example.ipynb](https://github.com/yjgeno/GenKI/blob/master/notebook/Example.ipynb)

## Citation

If you use GenKI in your research, please cite:

> Yang Y, Wang M, Ni P, Zhong J. *GenKI: Virtual gene knockout inference with variational graph autoencoder*. Nucleic Acids Research, 2023. https://doi.org/10.1093/nar/gkad450
