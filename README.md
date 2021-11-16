## Introduction

This repository contains the source code for the publications [GRAND: Graph Neural Diffusion](https://icml.cc/virtual/2021/poster/8889) and [Beltrami Flow and Neural Diffusion on Graphs (BLEND)](https://arxiv.org/abs/2110.09443).
These approaches treat deep learning on graphs as a continuous diffusion process and Graph Neural
Networks (GNNs) as discretisations of an underlying PDE. In both models, the layer structure and
topology correspond to the discretisation choices
of temporal and spatial operators. Our approach allows a principled development of a broad new
class of GNNs that are able to address the common plights of graph learning models such as
depth, oversmoothing, and bottlenecks. Key to
the success of our models are stability with respect to perturbations in the data and this is addressed for both 
implicit and explicit discretisation schemes. We develop linear and nonlinear
versions of GRAND, which achieve competitive results on many standard graph benchmarks. BLEND is a non-Euclidean extension of GRAND that jointly evolves the feature and positional encodings of each node providing a principled means to perform graph rewiring.

## Running the experiments

### Requirements
Dependencies (with python >= 3.7):
Main dependencies are
torch==1.8.1
torch-cluster==1.5.9
torch-geometric==1.7.0
torch-scatter==2.0.6
torch-sparse==0.6.9
torch-spline-conv==1.2.1
torchdiffeq==0.2.1
Commands to install all the dependencies in a new conda environment
```
conda create --name grand python=3.7
conda activate grand

pip install ogb pykeops
pip install torch==1.8.1
pip install git+https://github.com/google-research/torchsde.git
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric
```

### Dataset and Preprocessing for GRAND (Graph Neural Diffusion)
create a root level ./data folder. This will be automatically populated the first time each experiment is run.

### Experiments for GRAND (Graph Neural Diffusion)
For example to run for Cora with random splits:
```
cd src
python run_GNN.py --dataset Cora 
```

### Dataset and Preprocessing for BLEND (Beltrami Flow and Neural Diffusion on Graphs)
create a root level ./data folder. This will be automatically populated the first time each experiment is run.
create a root level ./data/pos_encodings folder. If using DeepWalk or Hyperbollic type positional encodings these will need generating using the relevant generator scripts.

### Experiments for BLEND (Beltrami Flow and Neural Diffusion on Graphs)
**Please note that we are still merging the code from the Twitter private BLEND repository (as of Nov 16th 21) and currently only the DIGL positional encodings are supported. This will change shortly.**
For example to run for Cora with random splits:
```
cd src
python run_GNN.py --dataset Cora --beltrami
```

## Troubleshooting 

Most problems installing the dependencies are caused by Cuda version mismatches with pytorch geometric. We recommend checking your cuda and pytorch versions
```
nvcc --version
python -c "import torch; print(torch.__version__)"
```
and then following instructions here to install pytorch geometric
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

## Cite us
If you found this work useful, please consider citing our papers
```
@article
{chamberlain2021grand,
  title={GRAND: Graph Neural Diffusion},
  author={Chamberlain, Benjamin Paul and Rowbottom, James and Goronova, Maria and Webb, Stefan and Rossi, 
  Emanuele and Bronstein, Michael M},
  journal={Proceedings of the 38th International Conference on Machine Learning,
               (ICML) 2021, 18-24 July 2021, Virtual Event},
  year={2021}
}
```
and
```
@article
{chamberlain2021blend,
  title={Beltrami Flow and Neural Diffusion on Graphs},
  author={Chamberlain, Benjamin Paul and Rowbottom, James and Eynard, Davide and Di Giovanni, Francesco and Dong Xiaowen and Bronstein, Michael M},
  journal={Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) 2021, Virtual Event},
  year={2021}
}
```




## Security Issues?
Please report sensitive security issues via Twitter's bug-bounty program (https://hackerone.com/twitter) rather than GitHub.

