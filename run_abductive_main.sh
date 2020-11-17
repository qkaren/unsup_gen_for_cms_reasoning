#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 abductive_main.py \
    --pretrained_model gpt2-medium \
    --length 15 \
    --max_length 30 \
    --top_k 1 \
    --num_passes 20 \
    --num_backward_iters 20 \
    --stepsize 0.0003 \
    --mix_rate 0.88 \
    --temperature_first 1 \
    --temperature_backward 1 \
    --temperature_forward 1\
    --input_file ./data/abductive/small_data.json  \
    --output_dir ./output/abductive/
