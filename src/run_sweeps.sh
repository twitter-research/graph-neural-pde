i#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$(($i % 8)) wandb agent graph_neural_diffusion/test_time_dep2/$1 &
done
