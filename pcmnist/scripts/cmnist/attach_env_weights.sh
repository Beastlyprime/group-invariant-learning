#!/bin/bash
python datasets/multienv/attach_env_weights.py \
--save_dir /Bias-Mitigators/pcmnist/label-noise-0.25/MLP_IRM/erm-390/bs_49998_ep_500_lr_0.0001 \
--env_type blocking \
--n_clusters 2 4 \
--dataset_name cmnist \
--auto_select