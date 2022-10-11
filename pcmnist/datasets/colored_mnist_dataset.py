
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import json
from utils.data_utils import dict_collate_fn
from datasets.rng_state import RNG_STATE
import pickle

def make_environment(images, labels, e, p_noise):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability p_noise
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(p_noise, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()
    }

def make_color_and_patch(images, labels, p_color, p_patch, p_noise=0.25):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability p_noise
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(p_noise, len(labels)))
    # Assign a color based on the label; flip the color with probability p_color
    colors = torch_xor(labels, torch_bernoulli(p_color, len(labels)))
    # Assign a patch based on the label; flip the patch label with probability p_patch
    patches = torch_xor(labels, torch_bernoulli(p_patch, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    images = (images.float() / 255.)
    # Add a patch to the left top corner when patches == 1, else to the right bottom corner
    for i in range(len(images)):
        images[i, :, :3, :3] += patches[i]
        images[i, :, -3:, -3:] += (1.-patches[i])
    return {
        'images': images.cuda(),
        'labels': labels[:, None].cuda()
    }

def joint_envs(envs): # to in consistent with EIIL
    # assumes first two entries in envs list are the train sets to joined
    print('pooling envs')
    # pool all training envs (defined as each env in envs[:-1])
    joined_train_envs = dict()
    for k in envs[0].keys():
        joined_values = torch.cat((envs[0][k][:-1],
                                   envs[1][k][:-1]),
                                  0)
        joined_train_envs[k] = joined_values
    print('size of pooled envs: %d' % len(joined_train_envs['images']))
    return joined_train_envs

class CMNISTDataset(Dataset):
    def __init__(self, original_set, group_ix=None, weight=None):
        self.set = original_set
        self.group_ix = group_ix if group_ix is not None else None
        self.weight = weight if weight is not None else None

    def __getitem__(self, index):
        item_data = {}
        item_data['dataset_ix'] = index
        item_data['x'] = self.set['images'][index]
        item_data['y'] = self.set['labels'][index][0]
        item_data['group_ix'] = self.group_ix[index] if self.group_ix is not None else 0
        if self.weight is not None:
            item_data['weight'] = self.weight[index]
        return item_data

    def _get_true_group_and_weight_scill(self):
        self.group_ix = (self.set['colors'] == 1) & (self.set['patches'] == 0).long()
        self.group_ix += 2 * ((self.set['colors'] == 0) & (self.set['patches'] == 1)).long()
        self.group_ix += 3 * ((self.set['colors'] == 1) & (self.set['patches'] == 1)).long()
        self.group_ix = self.group_ix.cuda()
        labels = self.set['labels'].reshape(-1,)
        self.weight = torch.ones(len(self.group_ix))
        for i in range(4):
            index_0 = ((labels == 0) & (self.group_ix == i))
            index_1 = ((labels == 1) & (self.group_ix == i))
            p_0 = index_0.sum() / (self.group_ix == i).sum()
            p_1 = 1. - p_0
            self.weight[index_0] = 1. / p_0
            self.weight[index_1] = 1. / p_1
        self.weight = self.weight.cuda()
    
    def _get_noisy_scill(self, deviation: float):
        self.group_ix = (self.set['colors'] == 1) & (self.set['patches'] == 0).long()
        self.group_ix += 2 * ((self.set['colors'] == 0) & (self.set['patches'] == 1)).long()
        self.group_ix += 3 * ((self.set['colors'] == 1) & (self.set['patches'] == 1)).long()
        self.group_ix = self.group_ix.cuda()
        labels = self.set['labels'].reshape(-1,)
        self.weight = torch.ones(len(self.group_ix))
        for i in range(4):
            index_0 = ((labels == 0) & (self.group_ix == i))
            index_1 = ((labels == 1) & (self.group_ix == i))
            p_0 = index_0.sum() / (self.group_ix == i).sum()
            p_0 = p_0 / deviation
            p_1 = 1. - p_0
            self.weight[index_0] = 1. / p_0
            self.weight[index_1] = 1. / p_1
        self.weight = self.weight.cuda()

    def _get_true_group_eiil(self):
        labels = self.set['labels'].reshape(-1,)
        self.group_ix = (self.set['colors'] == 0) & (labels == 1).long()
        self.group_ix += (self.set['colors'] == 1) & (labels == 0).long()
        self.group_ix = self.group_ix.cuda()

    def _get_true_group_and_weight_eiil(self):
        labels = self.set['labels'].reshape(-1,)
        self.group_ix = (self.set['colors'] == 0) & (labels == 1).long()
        self.group_ix += (self.set['colors'] == 1) & (labels == 0).long()
        self.group_ix = self.group_ix.cuda()
        self.weight = torch.ones(len(self.group_ix))
        for i in range(2):
            index_0 = ((labels == 0) & (self.group_ix == i))
            index_1 = ((labels == 1) & (self.group_ix == i))
            p_0 = index_0.sum() / (self.group_ix == i).sum()
            p_1 = 1. - p_0
            self.weight[index_0] = 1. / p_0
            self.weight[index_1] = 1. / p_1
        self.weight = self.weight.cuda()

    def __len__(self):
        return len(self.set['labels'])

def create_cmnist_datasets(p_noise=0.25, env_file_name=None):
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:55000], mnist.targets[50000:55000])
    mnist_test = (mnist.data[55000:], mnist.targets[55000:])

    # rng_state = np.random.get_state()
    np.random.set_state(RNG_STATE)
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(RNG_STATE)
    np.random.shuffle(mnist_train[1].numpy())
    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2, p_noise), # 0,2,4, ...
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1, p_noise), # 1,3,5, ...
        make_environment(mnist_val[0], mnist_val[1], 0.15, p_noise),
        make_environment(mnist_test[0], mnist_test[1], 0.5, p_noise) # not worst case!
    ]

    train_data = joint_envs(envs)

    if env_file_name is not None:
        env_data = json.load(open(f'{env_file_name}.json'))
        if 'val_weight' in env_data.keys():
            train_set = CMNISTDataset(train_data, env_data['group_ix'], env_data['weight'])
            val_set = CMNISTDataset(envs[2], env_data['val_group_ix'], env_data['val_weight'])
        else:
            train_set = CMNISTDataset(train_data, env_data['group_ix'])
            val_set = CMNISTDataset(envs[2])
    else:
        train_set = CMNISTDataset(train_data)
        val_set = CMNISTDataset(envs[2])
    test_set = CMNISTDataset(envs[3])
    return train_set, val_set, test_set

def create_pcmnist_datasets(p_noise=0.25, env_file_name=None, seed=0, config=None):
    if config.data_dir is None:
        mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:55000], mnist.targets[50000:55000])
        mnist_oracle = (mnist.data[55000:55500], mnist.targets[55000:55500])
        mnist_test = (mnist.data[55500:], mnist.targets[55500:])
        np.random.seed(seed)
        rng_state = np.random.get_state()
        # np.random.set_state(RNG_STATE)
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        # np.random.set_state(RNG_STATE)
        np.random.shuffle(mnist_train[1].numpy())
        envs = [
            make_color_and_patch(mnist_train[0], mnist_train[1], 0.1, 0.3, p_noise),
            make_color_and_patch(mnist_val[0], mnist_val[1], 0.1, 0.3, p_noise),
            make_color_and_patch(mnist_test[0], mnist_test[1], 0.5, 0.5, p_noise),
            make_color_and_patch(mnist_oracle[0], mnist_oracle[1], 0.5, 0.5, p_noise)
        ]
        with open(os.path.join(config.save_dir, 'data_envs.pkl'), 'wb') as f:
            pickle.dump(envs, f)
    else:
        with open(os.path.join(config.data_dir, 'data_envs.pkl'), 'rb') as f:
            envs = pickle.load(f)

    if env_file_name is not None:
        env_data = json.load(open(f'{env_file_name}.json'))
        if 'weight' in env_data.keys():
            train_set = CMNISTDataset(envs[0], env_data['group_ix'], env_data['weight'])
        else:
            train_set = CMNISTDataset(envs[0], env_data['group_ix'])
        if 'val_weight' in env_data.keys():
            val_set = CMNISTDataset(envs[1], env_data['val_group_ix'], env_data['val_weight'])
        else:
            val_set = CMNISTDataset(envs[1])
    else:
        train_set = CMNISTDataset(envs[0])
        val_set = CMNISTDataset(envs[1])
        if 'noisy' in config.env_type:
            train_set._get_noisy_scill(config.deviation)
            val_set._get_noisy_scill(config.deviation)
            print('Label balance abl. study.')
        elif 'scill' in config.env_type:
            train_set._get_true_group_and_weight_scill()
            val_set._get_true_group_and_weight_scill()
        elif 'eiil' in config.env_type:
            train_set._get_true_group_eiil()
            val_set._get_true_group_and_weight_eiil()

    test_set = CMNISTDataset(envs[2])
    oracle_set = CMNISTDataset(envs[3])
    return train_set, val_set, test_set, oracle_set

def create_cmnist_dataloaders(config):
    # trainer needs: batch['dataset_ix'], batch['y'], batch['x'], batch['weight'], batch['group_ix']
    if config.dataset_name == 'cmnist':
        train_set, val_set, test_set, oracle_set = create_cmnist_datasets(config.p_noise, config.env_file_name)
    elif config.dataset_name == 'pcmnist':
        train_set, val_set, test_set, oracle_set = create_pcmnist_datasets(config.p_noise, config.env_file_name, config.random_seed, config)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=dict_collate_fn(), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=100, shuffle=False,
                            num_workers=config.num_workers, collate_fn=dict_collate_fn())
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dict_collate_fn())
    oracle_loader = DataLoader(oracle_set, batch_size=100, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dict_collate_fn())
    
    return {
        'Train': train_loader,
        'Test': {
            'Train': train_loader,
            'Val': val_loader,
            # 'Balanced Val': balanced_val_loader,
            'Test': test_loader,
            'Oracle': oracle_loader
        }
    }


if __name__ == "__main__":
    from collections import Counter
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    print(Counter(mnist.targets[:50000]))