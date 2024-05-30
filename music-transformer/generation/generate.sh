#!/bin/bash

env_name="music_transformer_env"

cd ..
source $env_name/bin/activate

CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES

# python3 -m generation.evaluate_samples
python3 -m generation.generate_sample
