import os
import copy
import numpy as np
import wandb
import pprint
os.environ["WANDB_CONSOLE"] = "off"


def set_if_null(option, attr_name, val):
    if not hasattr(option, attr_name) or getattr(option, attr_name) is None:
        setattr(option, attr_name, val)

def cmnist_experiments(orig_option, run):
    # Method-specific arguments are mostly defined in the bash files inside: scripts/
    # orig_option = copy.deepcopy(option)
    # wandb.init(project=f"{option.dataset_name}", entity="abc")

    # wandb.config = {
    #     "learning_rate": option.lr,
    #     "epochs": option.epochs,
    #     "grad_penalty_weight": option.grad_penalty_weight,
    #     "anneal_epochs": option.penalty_weight_ascend_iter_n,
    #     "l2_reg_weight": option.l2_reg_weight
    # }
    option = copy.deepcopy(orig_option)
    option.target_name = 'labels'
    option.loss_type = "BCEWithLogitsLoss"
    option.in_dims=14*14

    # Test epochs
    option.test_epochs = [e for e in
                          np.arange(option.epochs - 30 * option.test_every, option.epochs, option.test_every)]  # Due to instability, we average accuracies over the last 10 epochs
    save_every = option.save_model_every
    option.test_every = save_every  # We further test every saved epochs
    option.save_predictions_every = save_every

    # option_dict = vars(copy.deepcopy(option))
    
    # set wandb sweep config
    sweep_config = {
        'method': 'grid',
        'name': f'{option.env_type}-{option.trainer_name}-{option.random_seed}',
        'entity': "abc",
        # 'method': 'random',
        # 'early_terminate': {
        #     'type': 'hyperband',
        #     'max_iter': 40
        # }
    }
    parameters_dict = {
        "learning_rate": {
            'values': [1e-4, 5e-4, 1e-3, 5e-3]
        },
        "epochs": {
            'value': option.epochs
        },
        "grad_penalty_weight": {
            'values': [0.1, 1, 10, 100]
        },
        "anneal_epochs": {
            'values': [100, 300, 500, 700]
        },
        "method": {
            'value': option.trainer_name
        }
    }
    parameters_dict_erm_base = {
        "learning_rate": {
            'values': [1e-4, 5e-4, 1e-3, 5e-3]
        },
        "epochs": {
            'value': option.epochs
        },
        "grad_penalty_weight": {
            'value': 0
        },
        "anneal_epochs": {
            'value': 0
        }
    }
    parameters_dict_erm = {
        "learning_rate": {
            'value': 1e-4
        },
        "epochs": {
            'value': 100
        },
        "grad_penalty_weight": {
            'value': 0
        },
        "anneal_epochs": {
            'value': 0
        }
    }
    parameters_dict_test = {
        "learning_rate": {
            'values': [5e-4]
        },
        "epochs": {
            'value': option.epochs
        },
        "grad_penalty_weight": {
            'values': [10]
        },
        "anneal_epochs": {
            'values': [700]
        }
    }
    if option.env_type == 'ERM':
        sweep_config['parameters'] = parameters_dict_erm
    else:
        sweep_config['parameters'] = parameters_dict
    
    # for param in option_dict.keys():
    #     sweep_config[param] = {'value':option_dict[param]}
    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project=f"{option.dataset_name}-{option.env_type}")

    return option, sweep_id
    

def run_expt(orig_option, run_fn, expt_name=''):
    option = copy.deepcopy(orig_option)
    option.target_name = 'labels'
    option.loss_type = "BCEWithLogitsLoss"
    option.in_dims=14*14

    # Optimizer + Model
    set_if_null(option, 'optimizer_name', 'Adam')

    # Test epochs
    option.test_epochs = [e for e in
                          np.arange(option.epochs - 30 * option.test_every, option.epochs, option.test_every)]  # Due to instability, we average accuracies over the last 10 epochs
    save_every = option.save_model_every
    option.test_every = save_every  # We further test every saved epochs
    option.save_predictions_every = save_every

    run_fn(option)

