#!/bin/bash
export PYTHONUNBUFFERED=1
EPOCHS=800
LR=0.0001
L2_WEIGHT=0.001
# SEED=0
# LR=0.0004898
# L2_WEIGHT=0.00110794568
EXPT=cmnist_experiments
DATASET_NAME=pcmnist
BATCH_SIZE=49998
TEST_EVERY=60
MODEL=MLP_IRM
NUM_CLASSES=2
NEPB=1
HID_DIMS=390
P_NOISE=0.25
PENALTY_ANNEAL_ITERS=190
# ENVS=Kmeans-2
# MAX_GROUPS=2
# ENV_FILE=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/erm-390/bs_49998_ep_500_lr_0.0004898/kmeans_2_weight

# ENVS=blocking
# MAX_GROUPS=19
# ENV_FILE=/Bias-Mitigators/${DATASET_NAME}/label-noise-${P_NOISE}/${MODEL}/erm-390/bs_49998_ep_500_lr_0.0001/blocking-anova-${MAX_GROUPS}