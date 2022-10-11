import json
env_file_name = '/Bias-Mitigators/pcmnist/label-noise-0.25/SEED-2/MLP_IRM/erm-390/lr-0.0001-pw-0-ep-100-annl0/train_eiil_0_0.001_10000.json'
with open(env_file_name) as fid:
    env = json.load(fid)
print(env.keys())