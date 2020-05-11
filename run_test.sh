#!/bin/bash


CUDA_VISIBLE_DEVICES=$1 python3 main.py \
    --pretrained_model gpt2-large \
    --length 10 \
    --max_length 30 \
    --mask_top_k 40 \
    --top_k 1 \
    --sample \
    --num_passes 20 \
    --num_iterations 40 \
    --stepsize 0.003 \
    --mix_rate 0.6 \
    --temperature_first 1 \
    --temperature_backward 1 \
    --temperature_forward 1 \
    --temperature_o1 0.0005 \
    --perturb_o1 \
    --verbose \
    --o1_text "James still loves to fish even if he is alone.<|endoftext|>James loves to go fishing." \
    --o2_text "James still loves to fish even if he is alone."

    #--o1_text "The bird ate it.<|endoftext|>Jane was sitting on a bench at the park." \
    #--o2_text "The bird ate it."

    #--o1_text "It always wastes my time.<|endoftext|>I work at a place that believes in teamwork." \
    #--o2_text "It always wastes my time."

    #--o1_text "She was very grateful.<|endoftext|>Sally goes to a school with lots of other kids." \
    #--o2_text "She was very grateful."

    #--o1_text "He checked it but it was completely broken.<|endoftext|>Stephen was at a party." \
    #--o2_text "He checked it but it was completely broken."

    #--o1_text "Months later I was able to reach my goal.<|endoftext|>My goal is to leg lift over two hundred pounds at the gym." \
    #--o2_text "Months later I was able to reach my goal."


### ====== Backups ======


#CUDA_VISIBLE_DEVICES=2 python3 main_bak.py \
#    --pretrained_model gpt2-large \
#    --length 15 \
#    --top_k 1 \
#    --sample \
#    --num_passes 20 \
#    --num_iterations 20 \
#    --stepsize 0.003 \
#    --mix_rate 0.8 \
#    --temperature_first 1 \
#    --temperature_backward 0.9 \
#    --temperature_forward 1 \
#    --temperature_o1 0.0005 \
#    --perturb_o1 \
#    --o1_text "The bird ate it.<|endoftext|>Jane was sitting on a bench at the park." \
#    --o2_text "The bird ate it."

    #--o1_text "She was very grateful.<|endoftext|>Sally goes to a school with lots of other kids." \
    #--o2_text "She was very grateful."

    #--o1_text "But it was time to give up her search.<|endoftext|>Gina's pencils had been missing for 2 days." \
    #--o2_text "But it was time to give up her search."

    #--o1_text "Afterwards, he changed the water and put Nemo back into his tank.<|endoftext|>Johnny got a new pet fish for his birthday." \
    #--o2_text "Afterwards, he changed the water and put Nemo back into his tank."

    #--o1_text "It was an exhausting experience.<|endoftext|>My high school band went to New york." \
    #--o2_text "It was an exhausting experience."

    #--o1_text "James still loves to fish even if he is alone.<|endoftext|>James loves to go fishing." \
    #--o2_text "James still loves to fish even if he is alone."


# =====


#CUDA_VISIBLE_DEVICES=2 python3 main.py \
#    --pretrained_model gpt2-large \
#    --length 15 \
#    --max_length 25 \
#    --mask_top_k 0 \
#    --top_k 1 \
#    --sample \
#    --num_passes 20 \
#    --num_iterations 400 \
#    --stepsize 0.001 \
#    --mix_rate 0.8 \
#    --temperature_first 1 \
#    --temperature_backward 1 \
#    --temperature_forward 1 \
#    --temperature_o1 0.0005 \
#    --perturb_o1 \
#    --verbose \
#    --o1_text "It always wastes my time.<|endoftext|>I work at a place that believes in teamwork." \
#    --o2_text "It always wastes my time."

    #--o1_text "The bird ate it.<|endoftext|>Jane was sitting on a bench at the park." \
    #--o2_text "The bird ate it."

    #--o1_text "It always wastes my time.<|endoftext|>I work at a place that believes in teamwork." \
    #--o2_text "It always wastes my time."

    #--o1_text "But it was time to give up her search.<|endoftext|>Gina's pencils had been missing for 2 days." \
    #--o2_text "But it was time to give up her search."

    #--o1_text "The bird ate it.<|endoftext|>Jane was sitting on a bench at the park." \
    #--o2_text "The bird ate it."


# =======


#CUDA_VISIBLE_DEVICES=2 python3 main.py \
#    --pretrained_model gpt2-large \
#    --length 15 \
#    --mask_top_k 40 \
#    --top_k 1 \
#    --sample \
#    --num_passes 20 \
#    --num_iterations 20 \
#    --stepsize 0.1 \
#    --mix_rate 0.8 \
#    --temperature_first 10 \
#    --temperature_backward 1 \
#    --temperature_forward 1 \
#    --temperature_o1 0.0005 \
#    --perturb_o1 \
#    --verbose \
#    --seed 0 \
#    --o1_text "The bird ate it.<|endoftext|>Jane was sitting on a bench at the park." \
#    --o2_text "The bird ate it."

    #--o1_text "It always wastes my time.<|endoftext|>I work at a place that believes in teamwork." \
    #--o2_text "It always wastes my time."

    #--o1_text "James still loves to fish even if he is alone.<|endoftext|>James loves to go fishing." \
    #--o2_text "James still loves to fish even if he is alone."

    #--o1_text "Tina was very happy on her birthday.<|endoftext|>My daughter's old college roommate Tina turned 30 last September." \
    #--o2_text "Tina was very happy on her birthday."

    #--o1_text "It was an exhausting experience.<|endoftext|>My high school band went to New york." \
    #--o2_text "It was an exhausting experience."

    #--o1_text "She took a sip and immediately spit it out, hating the taste.<|endoftext|>Cheyanne was at Easter brunch." \
    #--o2_text "She took a sip and immediately spit it out, hating the taste."


    #--o1_text "Afterwards, he changed the water and put Nemo back into his tank.<|endoftext|>Johnny got a new pet fish for his birthday." \
    #--o2_text "Afterwards, he changed the water and put Nemo back into his tank."


