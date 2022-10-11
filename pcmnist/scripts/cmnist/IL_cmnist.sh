#!/bin/bash
source common-cmnist.sh
set -e

# if [ ${ENVS} == Kmeans-12 ]
# then
# ENV_FILE=/Bias-Mitigators/cmnist/label-noise-0.25/erm-390/bs_30000_ep_301/train_kmeans_3_weight.json
# fi
# 'PGITrainer' 'IRMv1Trainer' 191257 'PGITrainer' 'VRExTrainer'
# 1e-4 2e-4

for trainer_name in 'IRMv1Trainer'; do
	for lr in 5e-4; do
		for anneal_ep in 100; do
			SAVE_DIR=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/${MODEL}_${HID_DIMS}/${ENVS}-${MAX_GROUPS}/${trainer_name}/
			if [ ! -f "$FILE" ]; then
				python main.py \
				--weight_decay 0 \
				--trainer_name ${trainer_name} \
				--grad_penalty_weight 1 \
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