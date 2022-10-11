import argparse
from numpy import random
import torch
from torch import nn, optim, autograd
from torch.distributed.distributed_c10d import group
from tqdm import tqdm
import json
import numpy as np

SEED = 1010
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-prediction-file", type=str, required=True)
    parser.add_argument("--save-file", type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--add_weights', action='store_true')
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
    assert "logits" in prediction or "log_probs" in prediction
    import numpy as np
    if "logits" in prediction:
        logits = torch.tensor(prediction['logits']).float().cuda()
        loss = nn.CrossEntropyLoss(reduction='none')(logits * scale, labels)
        N = logits.shape[0]
    else:
        log_probs = torch.tensor(prediction["log_probs"]).float().cuda()
        loss = nn.NLLLoss(reduction='none')(log_probs * scale, labels)
        N = log_probs.shape[0]

    env_w = torch.randn(N).cuda().requires_grad_()
    # env_w = torch.zeros(N).float().cuda().requires_grad_()
    optimizer = optim.Adam([env_w], lr=config.lr)

    print('learning soft environment assignments')
    for i in tqdm(range(config.n_steps)):
        # penalty for env a
        lossa = (loss.squeeze() * env_w.sigmoid()).mean()
        grada = autograd.grad(lossa, [scale], create_graph=True)[0]
        penaltya = torch.sum(grada**2)
        # penalty for env b
        lossb = (loss.squeeze() * (1-(env_w.sigmoid()))).mean()
        gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
        penaltyb = torch.sum(gradb**2)
        # negate
        npenalty = - torch.stack([penaltya, penaltyb]).mean()

        optimizer.zero_grad()
        npenalty.backward(retain_graph=True)
        optimizer.step()

    # split envs based on env_w threshold
    print('Environment W')
    hist = np.histogram(env_w.sigmoid().detach().cpu().numpy())
    print(hist[0])
    print(hist[1])
    Env_counts = [(torch.arange(len(env_w))[env_w.sigmoid()>0.5]).shape[0], (torch.arange(len(env_w))[env_w.sigmoid()<=0.5]).shape[0]]
    
    print('Env 1 Count', Env_counts[0], 'Env 2 Count', Env_counts[1])
    output = {"env_w": env_w.sigmoid().detach().cpu().numpy().tolist()}
    env_ids = (env_w.sigmoid() > 0.5).detach().cpu().numpy().tolist()
    # print((env_w > 0.5).detach().cpu().numpy().sum())
    groups = [1 if idx else 0 for idx in env_ids]
    output["group"] = groups
    if config.add_weights:
        labels = labels.cpu()
        unique_label_num = len(set(labels))
        env_weights = np.zeros((2, unique_label_num))
        env_0_labels = np.array(labels)[np.array(groups) == 0]
        env_1_labels = np.array(labels)[np.array(groups) == 1]
        for i in range(unique_label_num):
            if (env_0_labels == i).sum() > 0:
                env_weights[0, i] = Env_counts[0] / float((env_0_labels == i).sum())
            if (env_1_labels == i).sum() > 0:
                env_weights[1, i] = Env_counts[1] / float((env_1_labels == i).sum())
        env_total_weights = env_weights.sum(axis=1)
        for r in range(2):
            env_weights[r, :] = env_weights[r, :] / env_total_weights[r]
        output['weight'] = [env_weights[groups[i], labels[i]] for i in range(len(groups))]

    with open(config.save_file, "w") as f:
        json.dump(output, f) 
    
    


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    eiil(config)