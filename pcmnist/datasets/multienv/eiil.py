import argparse
from numpy import random
import torch
from torch import nn, optim, autograd
from torch.distributed.distributed_c10d import group
from tqdm import tqdm
import json
import numpy as np
from collections import Counter
import os

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def nll(logits, y, reduction='mean'):
  return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-prediction-file", type=str, required=True)
    parser.add_argument("--save-file", type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=10000)
    return parser

def eiil(config):
    print('Flags:')
    for k,v in sorted(vars(config).items()):
        print("\t{}: {}".format(k, v))
    
    # read prediction file 
    with open(config.pretrained_model_prediction_file) as f:
        prediction = json.load(f)
    scale = torch.tensor(1.).cuda().requires_grad_()
    labels = torch.tensor(prediction['y']).long().cuda()
    assert "logits" in prediction or "log_probs" in prediction or "logits_1" in prediction
    if "logits" in prediction:
        logits = torch.tensor(prediction['logits']).float().cuda()
        loss = nn.CrossEntropyLoss(reduction='none')(logits * scale, labels)
        N = logits.shape[0]
    elif "logits_1" in prediction:
        logits = torch.tensor(prediction['logits_1']).float().cuda()
        labels = torch.tensor(prediction['y']).float().cuda()
        loss = nll(logits * scale, labels, reduction='none')
        N = len(logits)
    else:
        log_probs = torch.tensor(prediction["log_probs"]).float().cuda()
        loss = nn.NLLLoss(reduction='none')(log_probs * scale, labels)
        N = log_probs.shape[0]

        
    env_w = torch.randn(N).cuda().requires_grad_()
    optimizer = optim.Adam([env_w], lr=config.lr)

    print('learning soft environment assignments')
    for i in tqdm(range(config.n_steps)):
        # penalty for env a
        lossa = (loss.squeeze() * env_w.sigmoid()).mean()
        grada = autograd.grad(lossa, [scale], create_graph=True)[0]
        penaltya = torch.sum(grada**2)
        # penalty for env b
        lossb = (loss.squeeze() * (1-env_w.sigmoid())).mean()
        gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
        penaltyb = torch.sum(gradb**2)
        # negate
        npenalty = - torch.stack([penaltya, penaltyb]).mean()

        optimizer.zero_grad()
        npenalty.backward(retain_graph=True)
        optimizer.step()
    
    print('Final AED Loss: %.8f' % npenalty.cpu().item())
    print("Scale: ", scale)

    # split envs based on env_w threshold
    print('Environment W')
    hist = np.histogram(env_w.sigmoid().detach().cpu().numpy())
    print(hist[0])
    print(hist[1])
    
    print('Env 1 Count', (torch.arange(len(env_w))[env_w.sigmoid()>0.5]).shape[0], 'Env 2 Count', (torch.arange(len(env_w))[env_w.sigmoid()<=0.5]).shape[0])
    output = {"env_w": env_w.sigmoid().detach().cpu().numpy().tolist()}
    env_ids = (env_w.sigmoid() > 0.5).detach().cpu().numpy().tolist()
    # print((env_w > 0.5).detach().cpu().numpy().sum())
    groups = [1 if idx else 0 for idx in env_ids]
    group_weight = np.ones(2)
    group_weight[1] = len(groups) / np.array(groups).sum() / 2.
    group_weight[0] = len(groups) / (len(groups) - np.array(groups).sum()) / 2.
    print("group weights: ", group_weight)
    labels = labels.long().detach().cpu().numpy()
    print("Env 1:", Counter(labels[(np.array(groups)==0)]))
    print("Env 2:", Counter(labels[(np.array(groups)==1)]))

    if config.is_val:
        with open(config.save_file, "r") as f:
            train_ei = json.load(f)
        val_weight = [group_weight[idx] for idx in groups]
        print("Examples of val_weight: ", val_weight[:10])
        train_ei['val_group_ix'] = groups
        train_ei['val_weight'] = val_weight
        with open(config.save_file, "w") as f:
            json.dump(train_ei, f)
        print("Results saved to: ", config.save_file)

    else:
        output["group_ix"] = groups
        with open(config.save_file, "w") as f:
            json.dump(output, f) 
    
    


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    eiil(config)