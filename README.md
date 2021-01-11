## Introduction

We provide a new perspective on graph neural networks (GNNs) by formulating them as 
discretizations of the diffusion PDE. This framework provides a rich class of models 
that encompasses various forms of graph rewiring and GNNs with both explicit and implicit layers


## Running the experiments

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing

create a root level data folder. This will be automatically populated the first time each experiment is run.

python run_GNN --dataset Cora

## Security Issues?
Please report sensitive security issues via Twitter's bug-bounty program (https://hackerone.com/twitter) rather than GitHub.

