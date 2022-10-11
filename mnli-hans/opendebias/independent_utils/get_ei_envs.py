import os
import json
import numpy as np
from collections import Counter
from eiil import eiil
class eiil_config(object):
    def __init__(self, lr, steps, save_dir, probs_file) -> None:
        self.lr = lr
        self.n_steps = steps
        self.save_file = os.path.join(save_dir, "val-eiil-1010.json")
        self.pretrained_model_prediction_file = probs_file
        self.add_weights = True

# HANS bias-only outputs on the validation set of MNLI
with open('/data/bias/mnli-hans/mind-trade-bias/dev.json') as f:
    val_bias = json.load(f)
print(val_bias.keys())

val_probs_file = "/data/bias/mnli-hans/mind-trade-bias/dev.json"
save_dir = "/data/bias/mnli-hans/mind-trade-bias"
ei_config = eiil_config(0.001, 10000, save_dir, val_probs_file)
eiil(ei_config)