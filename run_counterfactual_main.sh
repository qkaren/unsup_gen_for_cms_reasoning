#!/bin/bash

for num_passes in 5 10
do
  for num_backward_iters in 5 8 10 15
  do
    CUDA_VISIBLE_DEVICES=0 python3 counterfactual_main.py \
        --pretrained_model gpt2-medium \
        --mix_rate 0.92 \
        --temperature_forward 1 \
        --top_k 1 \
        --stepsize 0.0004 \
        --num_backward_iters ${num_backward_iters} \
        --num_passes ${num_passes} \
        --input_file ./data/counterfactual/small_data.json \
        --output_dir ./output/counterfactual/ \
        --verbose
  done
done
