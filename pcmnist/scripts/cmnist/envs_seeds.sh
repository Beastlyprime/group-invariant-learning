#!/bin/bash
# --------------------------------------------
# Get the EI and BMoS envs under a given set of seed.
# --------------------------------------------
source common-cmnist.sh
set -e
# source scripts/cmnist/erm_cmnist.sh
for seed in 0 1 2
do
    SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/SEED-${seed}/${MODEL}/erm-${HID_DIMS}/
    DATA_DIR=${SAVE_DIR}
    SAVE_DIR=${SAVE_DIR}lr-0.0001-pw-0-ep-100-annl0/

    python datasets/multienv/attach_env_weights.py \
    --save_dir ${SAVE_DIR} \
    --env_type eiil \
    --n_clusters 2 4 \
    --dataset_name cmnist \
    --auto_select

    python datasets/multienv/attach_env_weights.py \
    --save_dir ${SAVE_DIR} \
    --env_type blocking \
    --n_clusters 2 4 \
    --dataset_name cmnist \
    --auto_select
done