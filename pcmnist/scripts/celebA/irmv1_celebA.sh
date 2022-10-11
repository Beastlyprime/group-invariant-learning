#!/bin/bash
source common-celebA.sh
set -e
ROOT=/cache/results/celebA

TRAINER_NAME='IRMv1Trainer'
python main_celeba.py \
--lr 1e-4 \
--weight_decay 0 \
--expt_type celebA_experiments \
--trainer_name ${TRAINER_NAME} \
--grad_penalty_weight 1 \
--num_envs_per_batch 2 \
--epochs 50 \
--penalty_weight_ascend_iter_n 5 \
--data_dir /cache/celeba \
--model_name ${MODEL} \
--l2_reg_weight 0.001 \
--root_dir ${ROOT}
