#!/bin/bash
source common-cmnist.sh
set -e
# ---------------------------------------------
# Run experiments with a given seed and trainer
# ---------------------------------------------
SEED=0
SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/SEED-${SEED}/${MODEL}/erm-${HID_DIMS}/
DATA_DIR=${SAVE_DIR}

# 'VRExTrainer' 'PGITrainer' 'IRMv1Trainer'
trainer_name='IRMv1Trainer'
for WEIGHT in 0.1 1 10 100; do
    for LR in 1e-4 5e-4 1e-3 5e-3; do
        ENVS=group_truth_scill # group_truth_eiil # 
        MAX_GROUPS=4 # 2
        SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/SEED-${SEED}/${MODEL}_${HID_DIMS}/${ENVS}-${MAX_GROUPS}/${trainer_name}/
        python main_local.py \
        --weight_decay 0 \
        --trainer_name ${trainer_name} \
        --grad_penalty_weight ${WEIGHT} \
        --save_dir ${SAVE_DIR} \
        --data_dir ${DATA_DIR} \
        --random_seed ${SEED} \
        --lr 1e-4 \
        --expt_type ${EXPT} \
        --dataset_name ${DATASET_NAME} \
        --batch_size ${BATCH_SIZE} \
        --num_envs_per_batch ${NEPB} \
        --max_groups ${MAX_GROUPS} \
        --model_name ${MODEL} \
        --hid_dims ${HID_DIMS} \
        --num_classes ${NUM_CLASSES} \
        --epochs ${EPOCHS} \
        --test_every ${TEST_EVERY} \
        --env_file_name ${BMoS_ENV} \
        --penalty_weight_ascend_iter_n 100 \
        --save_model_every 8000 \
        --l2_reg_weight ${L2_WEIGHT} \
        --env_type ${ENVS} \
        --enable_tev
    done
done