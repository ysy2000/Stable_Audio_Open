# -*- coding: utf-8 -*-
"""Stable Audio Open
"""

# from huggingface_hub import login
# login(token="")

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


prompt = str(sys.argv[1])
output_dir = "/output/"
# os.makedirs(output_dir, exist_ok=True)

# Download model
model, model_config = get_pretrained_model('stabilityai/stable-audio-open-1.0')
sample_rate = model_config['sample_rate']
print(f'sample_rate: {sample_rate}')
sample_size = model_config['sample_size']
print(f'sample_size: {sample_size}')

model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    "prompt": prompt,
    "seconds_start": 0, 
    "seconds_total": 24
}]

# Generate stereo audio
# 378925029(ori) , 1600337421(mine)
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

num_samples = int(24 * sample_rate)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")
output = output[:,:num_samples]

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save(f"{prompt}.wav", output, sample_rate)
