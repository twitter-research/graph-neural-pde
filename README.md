![example workflow](https://github.com/twitter-research/graph-neural-pde/actions/workflows/python-package.yml/badge.svg)

![Cora_animation_16](https://user-images.githubusercontent.com/5874124/143270624-265c2d01-39ca-488c-b118-b68f876dfbfa.gif)

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
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric
```

### Troubleshooting

There is a bug in pandas==1.3.1 that could produce the error ImportError: cannot import name 'DtypeObj' from 'pandas._typing'
If encountered, then the fix is 
pip install pandas==1.3.0 -U

## GRAND (Graph Neural Diffusion)

### Dataset and Preprocessing
create a root level folder
```
./data
```
This will be automatically populated the first time each experiment is run.

### Experiments
For example to run for Cora with random splits:
```
cd src
python run_GNN.py --dataset Cora 
```

## BLEND (Beltrami Flow and Neural Diffusion on Graphs)

### Dataset and Preprocessing

Create a root level 
```
./data folder
``` 
This will be automatically populated the first time each experiment is run.
create a root level folder
```
./data/pos_encodings
```
DIGL positional encodings will build automatically and populate this folder, but DeepWalk or Hyperbollic positional encodings will need generating using the relevant generator scripts or downloading. We include a shell script (warning: it's slow) to generate them: 
```
sh deepwalk_gen.sh
```
then create symlinks to them with 
```
python deepwalk_gen_symlinks.py
```
Alternatively, we also provide precomputed positional encodings [here](https://www.dropbox.com/sh/wfktgbfiueikcp0/AABrIjyhR6Yi4EcirnryRXjja?dl=0)
Specifically, the positional encodings required to run the default settings for Citeseer, Computers, Phota and ogbn-arxiv are
- [Citeseer](https://www.dropbox.com/sh/wfktgbfiueikcp0/AAB9HypMFO3QCeDFojRYuQoDa/Citeseer_DW64.pkl?dl=0)
- [Computers](https://www.dropbox.com/sh/wfktgbfiueikcp0/AAD_evlqcwQFLL6MVyGeiKiha/Computers_DW128.pkl?dl=0)
- [Photo](https://www.dropbox.com/sh/wfktgbfiueikcp0/AAAAhsxAcHWB5OGTHLNMXR5-a/Photo_DW128.pkl?dl=0)
- [ogbn-arxiv](https://www.dropbox.com/sh/wfktgbfiueikcp0/AADcRPI5pLrx3iUvUjGBcqD0a/ogbn-arxiv_DW64.pkl?dl=0)

Download them and place into
```
./data/pos_encodings
```

### Experiments
 
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

