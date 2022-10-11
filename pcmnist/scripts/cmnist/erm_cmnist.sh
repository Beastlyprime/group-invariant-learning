#!/bin/bash
source common-cmnist.sh
set -e

SEED=2
TRAINER_NAME='BaseTrainer'
env_type='ERM'
wd=0
SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/SEED-${SEED}/${MODEL}/erm-${HID_DIMS}/
# SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/SEED-${SEED}/${MODEL}/erm-baseline/

python main.py \
--trainer_name ${TRAINER_NAME} \
--expt_type ${EXPT} \
--dataset_name ${DATASET_NAME} \
--grad_penalty_weight 0 \
--random_seed ${SEED} \
--p_noise ${P_NOISE} \
--model_name ${MODEL} \
--hid_dims ${HID_DIMS} \
--num_classes ${NUM_CLASSES} \
--batch_size ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--test_every ${TEST_EVERY} \
--lr ${LR} \
--max_groups 1 \
--num_envs_per_batch 1 \
--weight_decay ${wd} \
--save_dir ${SAVE_DIR} \
--save_model_every 1000 \
--l2_reg_weight ${L2_WEIGHT} \
--env_type ${env_type}
# --save_model_every 100 \

# python show_results.py --file ${SAVE_DIR}

# python datasets/multienv/attach_env_weights.py \
# --save_dir ${SAVE_DIR} \
# --env_type eiil \
# --dataset_name cmnist