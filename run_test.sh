#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --pretrained_model gpt2-medium \
    --length 15 \
    --top_k 1 \
    --sample \
    --num_passes 5 \
    --num_iterations 20 \
    --stepsize 10. \
    --mix_rate 0.6 \
    --temperature_first 0.6 \
    --temperature_backward 3 \
    --temperature_forward 0.5 \
    --temperature_o1 0.01 \
    --perturb_o1 \
    --o1_text "Stephen was at a party." \
    --o2_text "He checked it but it was completely broken."
    # --o1_text "Missy love her chocolate." \
    # --o2_text "Missy got the biggest chocolate bar and became super happy."

    #--o1_text "Johnny got a new pet fish for his birthday." \
    #--o2_text "Afterwards, he changed the water and put Nemo back into his tank."
