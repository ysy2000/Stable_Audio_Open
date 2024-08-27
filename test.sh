#!/bin/bash

python3 train.py \
--dataset-config /home/rxkmmm/Stable_Audio_Open/dataset.json \
--model-config /home/rxkmmm/Stable_Audio_Open/model_config.json \
--name stable_audio_open_finetune \
--save-dir /home/rxkmmm/Stable_Audio_Open/checkpoints \
--checkpoint-every 1000 \
--precision 16-mixed \
--seed 128 \
--batch-size 32 \
--pretrained-ckpt-path /home/rxkmmm/mnt/stable_audio_open/model.ckpt

# --num-gpus 2