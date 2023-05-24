# GenKI (Gene Knock-out Inference)
A VGAE (Variational Graph Auto-Encoder) based model to learn perturbation using scRNA-seq data. <br>
<br/>
<p align="center">
    <img src="logo.jpg" alt="drawing" width="300"/>
</p>
<br/>

### Install dependencies
Fist install dependencies of GenKI with `conda`:
```shell
conda env create -f environment.yml
conda activate ogenki
```
Install `pytorch-geometric` following the document:<br>
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
<br/>
<br/>

### Install GenKI (from source)
```shell
git clone https://github.com/yjgeno/GenKI.git
cd GenKI
pip install .
```
<br/>
or

```shell
pip install git+https://github.com/yjgeno/GenKI.git
```
<br/>

### Usages
#### Quick Start

