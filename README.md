# GenKI (Gene Knock-out Inference)
A VGAE (Variational Graph Auto-Encoder) based model to learn perturbation using scRNA-seq data. <br>
<span style="color:red;">New!</span> Data has been added. <br>
[Paper](https://doi.org/10.1093/nar/gkad450)
<br/>
<p align="center">
    <img src="logo.jpg" alt="drawing" width="300"/>
</p>
<br/>

### Prerequisites
Before installing GenKI, install [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) matching your CUDA version.

### Install GenKI with `pip` (PyPI):
```shell
pip install GenKI
```

Or install from source:
```shell
pip install git+https://github.com/yjgeno/GenKI.git
```

Or clone and install manually:
```shell
git clone https://github.com/yjgeno/GenKI.git
cd GenKI
pip install .
```

Alternatively, use `conda` to set up the full environment:
```shell
conda env create -f environment.yml
conda activate ogenki
```
<br/>

#### Tutorial
Virtual KO experiment:<br> https://github.com/yjgeno/GenKI/blob/master/notebook/Example.ipynb <br>
