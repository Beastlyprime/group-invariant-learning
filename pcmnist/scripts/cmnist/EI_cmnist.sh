#!/bin/bash
source common-cmnist.sh
set -e

ENVS=EI-10000
MAX_GROUPS=2
ENV_FILE=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/${MODEL}/erm-390/bs_49998_ep_500_lr_0.0001/train_eiil_0_0.001_10000
EPOCHS=500

# 191257 'PGITrainer' 'VRExTrainer' 'IRMv1Trainer'
for trainer_name in 'CMMDTrainer'; do 
	for grad_penalty_weight in 10; do
		for lr in 5e-4; do
			for anneal_ep in 190; do
				SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/${MODEL}_${HID_DIMS}/${ENVS}/${trainer_name}/lr_${lr}_l2_${L2_WEIGHT}/
				if [ ! -f "$FILE" ]; then
					python main.py \
					--weight_decay 0 \
					--trainer_name ${trainer_name} \
					--grad_penalty_weight ${grad_penalty_weight} \
					--save_dir ${SAVE_DIR} \
					--random_seed ${SEED} \
					--lr ${lr} \
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
					--env_file_name ${ENV_FILE} \
					--penalty_weight_ascend_iter_n ${anneal_ep} \
					--save_model_every 8000 \
					--l2_reg_weight ${L2_WEIGHT} \
					--env_type ${ENVS} \
					--enable_tev

					# python show_results.py \
					# --file ${SAVE_DIR}
				fi
			done
		done
	done
done