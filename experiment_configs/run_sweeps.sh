#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$(($i % 8)) wandb agent bchamberlain/graph-neural-pde-src/$1 &
done