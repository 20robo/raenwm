#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: bash scripts/run_eval.sh [time|rollout] [gt|infer|eval|all] [DATASET] [CKPT_PATH]"
    exit 1
fi

EVAL_MODE=$1
STEP=$2
DATASET=$3
CKPT_PATH=$4

RESULTS_FOLDER="results"
EXP_DIR="${RESULTS_FOLDER}/exp_dir_euler"

echo "========================================"
echo "Starting Evaluation Pipeline"
echo "Mode: ${EVAL_MODE}"
echo "Step: ${STEP}"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CKPT_PATH}"
echo "========================================"

# Define function for Ground-Truth preparation
run_gt() {
    echo "--- Running Ground-Truth Preparation ---"
    if [ "$EVAL_MODE" = "time" ]; then
_workers 12 --input_fps 4 --checkpoint_path ${CKPT_PATH} --sampling_method euler --num_steps 50 --eval_type time --output_dir ${RESULTS_FOLDER} --gt 1 --num_sec_eval 5 --eval_min_dist_cat -64 --eval_max_dist_cat 64 --eval_len_traj_pred 64
    elif        python infer.py --exp config/raenwm.yaml --datasets ${DATASET} --batch_size 32 --num [ "$EVAL_MODE" = "rollout" ]; then
        python infer.py --exp config/raenwm.yaml --datasets ${DATASET} --batch_size 32 --num_workers 12 --input_fps 4 --checkpoint_path ${CKPT_PATH} --sampling_method euler --num_steps 50 --eval_type rollout --output_dir ${RESULTS_FOLDER} --gt 1 --rollout_fps_values 4 --num_sec_eval 5 --eval_min_dist_cat -64 --eval_max_dist_cat 64 --eval_len_traj_pred 64
    fi
}

# Define function for Inference
run_infer() {
    echo "--- Running Future Frame Prediction ---"
    if [ "$EVAL_MODE" = "time" ]; then
        python infer.py --exp config/raenwm.yaml --datasets ${DATASET} --batch_size 32 --num_workers 12 --input_fps 4 --checkpoint_path ${CKPT_PATH} --sampling_method euler --num_steps 50 --eval_type time --output_dir ${RESULTS_FOLDER} --gt 0 --num_sec_eval 5 --eval_min_dist_cat -64 --eval_max_dist_cat 64 --eval_len_traj_pred 64
    elif [ "$EVAL_MODE" = "rollout" ]; then
        python infer.py --exp config/raenwm.yaml --datasets ${DATASET} --batch_size 32 --num_workers 12 --input_fps 4 --checkpoint_path ${CKPT_PATH} --sampling_method euler --num_steps 50 --eval_type rollout --output_dir ${RESULTS_FOLDER} --gt 0 --rollout_fps_values 4 --num_sec_eval 5 --eval_min_dist_cat -64 --eval_max_dist_cat 64 --eval_len_traj_pred 64
    fi
}

# Define function for Evaluation
run_eval() {
    echo "--- Running Metrics Evaluation ---"
    if [ "$EVAL_MODE" = "time" ]; then
        python evaluate.py --datasets ${DATASET} --gt_dir ${RESULTS_FOLDER}/gt --exp_dir ${EXP_DIR} --eval_types time --num_sec_eval 5 --batch_size 16
    elif [ "$EVAL_MODE" = "rollout" ]; then
        python evaluate.py --datasets ${DATASET} --gt_dir ${RESULTS_FOLDER}/gt --exp_dir ${EXP_DIR} --eval_types rollout --rollout_fps_values 4 --input_fps 4 --num_sec_eval 5 --batch_size 16
    fi
}

# Execute based on the STEP argument
case $STEP in
    gt)
        run_gt
        ;;
    infer)
        run_infer
        ;;
    eval)
        run_eval
        ;;
    all)
        run_gt
        run_infer
        run_eval
        ;;
    *)
        echo "Error: Invalid step '${STEP}'. Must be 'gt', 'infer', 'eval', or 'all'."
        exit 1
        ;;
esac

echo "Evaluation pipeline finished successfully!"