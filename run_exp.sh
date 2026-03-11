#!/bin/bash
# Usage: ./run_exp.sh <config_name> <gpu_id>
# Example: ./run_exp.sh ep4_paper 0

CONFIG_NAME=$1
GPU_ID=${2:-0}

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: $0 <config_name> [gpu_id]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export DEBUG=false

# Run Hydra, adding experiments to config search path
python -m src.main \
    --config-dir experiments \
    --config-name "$CONFIG_NAME" \
    mode=test
