#!/bin/bash

root_dir="../"
hyps_dir="${root_dir}/output/counterfactual/"
output_dir="${hyps_dir}/ranking"

original_data_fname="${root_dir}/data/counterfactual/small_data.json"

CUDA_VISIBLE_DEVICES=0 \
python3 counterfactual_ranking.py \
  --hyps_dir=${hyps_dir} \
  --output_dir=${output_dir} \
  --align_to_original_data \
  --original_data_file ${original_data_fname}
