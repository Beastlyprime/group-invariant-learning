import os
import numpy as np
from blocking import blocking_envs_hans
import json

## HANS
with open('/data/bias/mnli-hans/mind-trade-bias/train.json') as f:
    train_bias = json.load(f)
print(train_bias.keys())
print(train_bias['index2token'])

# Transform bias to 2 class
train_bias_probs = np.exp(np.array(train_bias['log_probs']))
train_bias_logprobs_2dim = np.zeros((len(train_bias_probs), 2))
train_bias_probs_2dim = np.zeros((len(train_bias_probs), 2))
train_bias_probs_2dim[:, 0] = train_bias_probs[:, 0]
train_bias_probs_2dim[:, 1] = 1 - train_bias_probs[:, 0]
train_bias_logprobs_2dim = np.log(train_bias_probs_2dim)
train_bias_y_2c = np.where(np.array(train_bias['y'])==0, 0, 1) # in 2-class, 1: entailment, 0:others

save_folder = "/data/bias/mnli-hans/mind-trade-bias/envs/20.0/"
blocking_envs_hans(train_bias, train_bias_probs_2dim, train_bias_y_2c, 20.0, save_folder=save_folder, save=True)