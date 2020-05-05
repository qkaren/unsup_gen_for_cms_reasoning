#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 counter_main.py \
    --pretrained_model gpt2-large \
    --top_k 1 \
    --sample \
    --num_passes 10 \
    --num_iterations 30 \
    --stepsize 0.001 \
    --mix_rate 0.8 \
    --temperature_first 1 \
    --temperature_backward 1. \
    --temperature_forward 1. \
    --o1_text "Ned wanted to learn to play basketball. He hung out in the library studying English." \
    --o2_text "Soon he felt confident enough to shoot a few hoops himself. The team watched him play and they cheered. Ned's skills improved as he practiced."

    #--verbose \

    #--o1_text "I wanted a pet for my birthday. Only dogs are allowed in my apartment building." \
    #--o2_text "I was looking around on facebook and saw a mini pig. I went to pick her up. I drove home with the mini pig in my car."

    # --o1_text "Sam and John went out to play some ultimate Frisbee one day. Upon arrival at the field they found it deserted." \
    # --o2_text "Sam approached them and asked them to let him and John play as well. After a few minutes talk, they agreed and everyone played for a bit. Then they all went home."

    #--o1_text "My class went to the Everglades for our field trip. We ended up having to turn back due to severe weather, and never saw the Everglades." \
    #--o2_text "We also got the opportunity to travel in water. The bus ride home was long and boring. I was tired when I got home."

    #--o1_text "Ryan was called by his friend to skip work one day. But Ryan had an important project at work and went in to finish it." \
    #--o2_text "Ryan and his friend played with birds at the park all day. At the end of the day, they left the park and saw Ryan's boss. Ryan got fired."

    #--o1_text "Neil had been journeying through Asia. But he contracted malaria on a Thai island, and had to be flown home for treatment." \
    #--o2_text "Neil was so excited to see Australian culture. He was thrilled at the prospect of exotic animals and people! His favorite moment was when he got to feed a baby koala bear."

    #--o1_text "Pierre loved Halloween. He decided to be a werewolf this year." \
    #--o2_text "He got a black cape and white face paint. His fake teeth were uncomfortable but looked great. Pierre couldn’t wait to go trick or treating!"

