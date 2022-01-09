#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$(($i % 8)) wandb agent graph-neural-diffusion/graph-neural-pde-src/$1 &
done