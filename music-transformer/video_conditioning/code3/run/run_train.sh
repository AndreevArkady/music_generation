#!/bin/bash

cd ..
env_name="music_transformer_env"

source $env_name/bin/activate
CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES

python3 -m train -path_to_params "parameters.json" -path_to_model_params "model/model_params.json"
