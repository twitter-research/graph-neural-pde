## Introduction

We present Graph Neural Diffusion (GRAND)
that approaches deep learning on graphs as a continuous diffusion process and treats Graph Neural
Networks (GNNs) as discretisations of an underlying PDE. In our model, the layer structure and
topology correspond to the discretisation choices
of temporal and spatial operators. Our approach allows a principled development of a broad new
class of GNNs that are able to address the common plights of graph learning models such as
depth, oversmoothing, and bottlenecks. Key to
the success of our models are stability with respect to perturbations in the data and this is addressed for both 
implicit and explicit discretisation schemes. We develop linear and nonlinear
versions of GRAND, which achieve competitive results on many standard graph benchmarks.

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
To create the required environment run
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Dataset and Preprocessing
create a root level ./data folder. This will be automatically populated the first time each experiment is run.
For example to run for Cora:
```
cd src
python run_GNN.py --dataset Cora 
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
If you found this work useful, please consider citing
```
@article
{chamberlain2021grand,
  title={GRAND: Graph Neural Diffusion},
  author={Chamberlain, Benjamin Paul and Rowbottom, James and Goronova, Maria and Webb, Stefan and Rossi, 
  Emanuele and Bronstein, Michael M},
  journal={Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  year={2021}
}
```


## Security Issues?
Please report sensitive security issues via Twitter's bug-bounty program (https://hackerone.com/twitter) rather than GitHub.

