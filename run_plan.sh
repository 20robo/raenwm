#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash scripts/run_plan.sh [DATASET] [curve | line] [CKPT_PATH]"
    exit 1
fi

DATASET=$1
SAMPLER=$2
CKPT_PATH=$3
RESULTS_FOLDER="results_plan"

echo "========================================"
echo "Starting Planning (CEM)"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CKPT_PATH}"
echo "========================================"

python planning_eval.py \
  --exp config/raenwm.yaml \
  --datasets ${DATASET} \
  --rollout_stride 1 \
  --batch_size 24 \
  --num_samples 120 \
  --topk 3 \
  --num_workers 12 \
  --output_dir ${RESULTS_FOLDER} \
  --run_tag results_plan \
  --opt_steps 1 \
  --num_repeat_eval 1 \
  --traj_sampler ${SAMPLER} \
  --checkpoint_path ${CKPT_PATH} \
  --save_preds \
  --plot

echo "Planning finished successfully!"