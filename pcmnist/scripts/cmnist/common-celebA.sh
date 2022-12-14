#!/bin/bash
export PYTHONUNBUFFERED=1
EPOCHS=800
LR=0.0001
L2_WEIGHT=0.001
EXPT=celebA_experiments
DATASET_NAME=CelebA
BATCH_SIZE=128
TEST_EVERY=2
MODEL=ResNet18
NUM_CLASSES=2
NEPB=1
HID_DIMS=390
P_NOISE=0.25
PENALTY_ANNEAL_ITERS=190