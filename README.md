## Introduction

We propose a novel class of graph neural networks based on the discretized Beltrami flow, a non-Euclidean diffusion PDE. In our model, node features are supplemented with positional encodings derived from the graph topology and jointly evolved by the Beltrami flow, producing simultaneously continuous feature learning, topology evolution. The resulting model generalizes many popular graph neural networks
and achieves state-of-the-art results on several benchmarks.

## Running the experiments

### Requirements

Dependencies (with python >= 3.7):

To create the required environment run
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Dataset and Preprocessing

create a root level ./data folder. This will be automatically populated the first time each experiment is run.

in the case of using deep walk or hyperbolic positional encodings upload provided positional encodings to ./data/pos_encodings

For example to run for Cora:
```
python run_GNN.py --dataset Cora 
```
optional --random_split, to run on randomized data splits

## Security Issues?
Please report sensitive security issues via Twitter's bug-bounty program (https://hackerone.com/twitter) rather than GitHub.

