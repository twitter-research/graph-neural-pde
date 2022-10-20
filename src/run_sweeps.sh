#!/bin/bash

for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$(($i % 1)) wandb agent graph_neural_diffusion/planetoid_best/$1 &
done