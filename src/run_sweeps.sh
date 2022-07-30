i#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$(($i % 8)) wandb agent graph_neural_diffusion/nonlin_gcn_ablation/$1 &
done
