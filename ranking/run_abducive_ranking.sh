#!/bin/bash

root_dir="../"
hyps_dir="${root_dir}/output/abductive"

CUDA_VISIBLE_DEVICES=0 \
python3 abductive_ranking.py \
  --hyps_file "${hyps_dir}/abductive_output_np20_nbi20.json"  \
  --output_dir "${hyps_dir}/ranking/" \
  --original_data_file "${root_dir}/data/abductive/small_data.json"
