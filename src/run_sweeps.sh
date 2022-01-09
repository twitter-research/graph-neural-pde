#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$(($i % 8)) wandb agent graph_neural_diffusion/greed_sweeps/$1 &
done