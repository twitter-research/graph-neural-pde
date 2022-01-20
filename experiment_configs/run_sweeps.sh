#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$(($i % 8)) wandb agent bchamberlain/waveGNN-src_node_level/$1 &
done