#!/bin/bash
source common-cmnist.sh
set -e
# ---------------------------------------------
# Run experiments with a given seed and trainer
# ---------------------------------------------
SEED=1
SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/SEED-${SEED}/${MODEL}/erm-${HID_DIMS}/
DATA_DIR=${SAVE_DIR}
SAVE_DIR=${SAVE_DIR}lr-0.0001-pw-0-ep-100-annl0/
EIIL_ENV=${SAVE_DIR}/train_eiil_0_0.001_10000
# BMoS_ENV=$(ls ${SAVE_DIR}/blocking-anova-*)a
# BMoS_ENV_UW=$(ls ${SAVE_DIR}/unweight-blocking-*)
BMoS_ENV=$(ls ${SAVE_DIR}/blocking-ttest-9*)
BMoS_ENV=$(echo ${BMoS_ENV%%.json})
BMoS_ENV_UW=$(ls ${SAVE_DIR}/unweight-blocking-t-9*)
BMoS_ENV_UW=$(echo ${BMoS_ENV_UW%%.json})

# 'VRExTrainer' 'PGITrainer' 'IRMv1Trainer'
for trainer_name in 'CMMDTrainer'; do
    # bmos
    ENVS=blocking-10
    MAX_GROUPS=9
    SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/SEED-${SEED}/${MODEL}_${HID_DIMS}/${ENVS}-${MAX_GROUPS}/${trainer_name}/
    python main.py \
    --weight_decay 0 \
    --trainer_name ${trainer_name} \
    --grad_penalty_weight 1 \
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