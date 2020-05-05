#!/bin/bash

split_n=2

CUDA_VISIBLE_DEVICES=0 python3 main_batch1.py \
    --pretrained_model gpt2-large \
    --length 10 \
    --top_k 1 \
    --sample \
    --num_passes 20\
    --num_iterations 20 \
    --stepsize 0.003 \
    --mix_rate 0.8 \
    --temperature_first 1 \
    --temperature_backward 1 \
    --temperature_forward 1\
    --temperature_o1 0.0005 \
    --perturb_o1 \
    --anli_jsonl_file ./exp_abd/${split_n}_dev.jsonl  \
    --output_dir ./output_abd/${split_n}

# CUDA_VISIBLE_DEVICES=${split_n} python3 main_batch1.py \
#     --pretrained_model gpt2-large \
#     --length 17 \
#     --top_k 1 \
#     --sample \
#     --num_passes 20\
#     --num_iterations 20 \
#     --stepsize 0.003 \
#     --mix_rate 0.8 \
#     --temperature_first 1 \
#     --temperature_backward 1 \
#     --temperature_forward 1\
#     --temperature_o1 0.0005 \
#     --perturb_o1 \
#     --anli_jsonl_file ./exp_abd/${split_n}_dev.jsonl  \
#     --output_dir ./output_abd/${split_n}