# -*- coding: utf-8 -*-
import os
import random
import wandb
import pprint

import numpy as np
import torch
from torch.backends import cudnn

from option import get_option
from trainers import trainer_factory
from utils.trainer_utils import save_option, initialize_logger
import logging
from datasets import dataloader_factory
import json
from experiments.celebA_experiments import *
from experiments.gqa_experiments import *
from experiments.biased_mnist_experiments import *
from experiments.cmnist_experiments import *
# os.environ["WANDB_CONSOLE"] = "off"

option = {}

def backend_setting():
    global option
    # Initialize the expt_dir where all the results (predictions, checkpoints, logs, metrics) will be saved
    if option.expt_dir is None:
        option.expt_dir = os.path.join(option.save_dir, f'lr-{option.lr}-pw-{option.grad_penalty_weight}-ep-{option.epochs}-annl{option.penalty_weight_ascend_iter_n}')

    if not os.path.exists(option.expt_dir):
        os.makedirs(option.expt_dir)

    # Configure the logger
    initialize_logger(option.expt_dir)

    # Set the random seeds
    if option.random_seed is None:
        option.random_seed = random.randint(1, 10000)
    random.seed(option.random_seed)
    torch.manual_seed(option.random_seed)
    torch.cuda.manual_seed_all(option.random_seed)
    np.random.seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        logging.warn('GPU is available, but we are not using it!!!')

    if not torch.cuda.is_available() and option.cuda:
        option.cuda = False

    # Dataset specific settings
    set_if_null(option, 'bias_loss_gamma', 0.7)
    set_if_null(option, 'bias_ema_gamma', 0.7)

def set_if_null(option, attr_name, val):
    if not hasattr(option, attr_name) or getattr(option, attr_name) is None:
        setattr(option, attr_name, val)

def main():
    global option
    option = get_option()
    if option.project_name is None:
        option.project_name = option.dataset_name
    if option.expt_type is not None:
        new_option, sweep_id = eval(option.expt_type)(option, run_with_wandb)
        option = new_option
        wandb.agent(sweep_id, run_with_wandb)
        # wandb.agent(sweep_id, run_with_wandb, count=1)
    else:
        run(option)

def run(option):
    backend_setting(option)
    # neptune.create_experiment(name=option.trainer_name + "_" + option.expt_name,
    #                           params=option.__dict__)
    # neptune.append_tag(option.trainer_name + "_" + option.expt_name)
    data_loaders = dataloader_factory.build_dataloaders(option)
    if 'gqa' in option.dataset_name.lower():
        option.bias_variable_dims = option.num_groups
        option.num_bias_classes = option.num_groups

    save_option(option)
    logging.getLogger().info(json.dumps(option.__dict__, indent=4, sort_keys=True,
                                        default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>"))

    trainer = trainer_factory.build_trainer(option)
    trainer.train(data_loaders['Train'], data_loaders['Test'])

def run_with_wandb():
    with wandb.init():
        wb_config = wandb.config
        option.lr = wb_config.learning_rate
        option.epochs = wb_config.epochs
        option.grad_penalty_weight = wb_config.grad_penalty_weight
        option.penalty_weight_ascend_iter_n = wb_config.anneal_epochs
        backend_setting()
        if 'gqa' in option.dataset_name.lower():
            option.bias_variable_dims = option.num_groups
            option.num_bias_classes = option.num_groups
        save_option(option)

        # neptune.create_experiment(name=option.trainer_name + "_" + option.expt_name,
        #                           params=option.__dict__)
        # neptune.append_tag(option.trainer_name + "_" + option.expt_name)
        data_loaders = dataloader_factory.build_dataloaders(option)

        logging.getLogger().info(json.dumps(option.__dict__, indent=4, sort_keys=True,
                                            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>"))
        if 'cmnist' in option.dataset_name.lower():
            wandb.define_metric("Val_TEV", summary="max")
            wandb.define_metric("Val", summary="max")
            wandb.define_metric("Test", summary="max")
            wandb.define_metric("Oracle", summary="max")
        trainer = trainer_factory.build_trainer(option)
        trainer.train(data_loaders['Train'], data_loaders['Test'])



if __name__ == '__main__':
    main()
