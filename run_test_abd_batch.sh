#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 abd_main_batch.py \
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
    --anli_jsonl_file ./dev.jsonl  \
    --output_dir ./output_abd/
