# Graph Neural Networks as Gradient Flows

This repository is the official implementation of [Graph Neural Networks as Gradient Flows](https://nips.cc/).

## Requirements

Dependencies (with python >= 3.7): Main dependencies are torch==1.8.1 torch-cluster==1.5.9 torch-geometric==1.7.0 torch-scatter==2.0.6 torch-sparse==0.6.9 torch-spline-conv==1.2.1 torchdiffeq==0.2.1 
Commands to install all the dependencies in a new conda environment

```
conda create --name graff python=3.7
conda activate graff

pip install torch==1.8.1
pip install torchdiffeq -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric
```

## Training and Evaluation

To train the model(s) in the paper, run this command:

```train
python run_GNN.py --dataset chameleon --use_best_params --num_splits 1
```

## Results

Examples of our Gradient Flow Framework (GRAFF) achieve the following performance on the following semi-supervised node classification benchmarks:

|  | Texas | Wisconsin | Cornell | Film | Squirrel | Chameleon | Citeseer | Pubmed | Cora |
|---|---|---|---|---|---|---|---|---|---|
| Hom level | 0.11 | 0.21 | 0.3 | 0.22 | 0.22 | 0.23 | 0.74 | 0.8 | 0.81 |
| #Nodes | 183 | 251 | 183 | 7,600 | 5,201 | 2,277 | 3,327 | 18,717 | 2,708  |
| #Edges | 295 | 466 | 280 | 26,752 | 198,493 | 31,421 | 4,676 | 44,327 | 5,278  |
| #Classes | 5 | 5 | 5 | 5 | 5 | 5 | 7 | 3 | 6 |
| GRAFF(DD) | 88.38±4.53 | 87.45±2.94 | 83.24±6.49 | 36.09±0.81 | 54.52±1.37 | 71.08±1.75 | 76.92±1.7 | 88.95±0.52 | 87.61±0.97 |
| GRAFF(D) | 88.11±5.57 | 88.83±3.29 | 84.05±6.1 | 37.11±1.08 | 47.36±1.89 | 66.78±1.28 | 77.3±1.85 | 90.04±0.41 | 88.01±1.03 |
| GRAFF-timedep(DD) | 87.03±4.49 | 87.06±4.04 | 82.16±7.07 | 35.93±1.23 | 53.97±1.45 | 69.56±1.2 | 76.59±1.53 | 88.26±0.41 | 87.38±1.05 |
