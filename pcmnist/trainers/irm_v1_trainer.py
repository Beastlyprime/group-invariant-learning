# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# 10.20 updated - finish
# 11.03 update compute_weighted_loss the same to eiil
# 11.16 test group balanced penalty
# 11.18 remove "group balanced" -- it is already is

import logging
import random

import numpy as np
from torch.utils.data import Subset, DataLoader

from trainers.base_trainer import BaseTrainer
from utils.losses import *
from utils.penalty_weight_scheduler import PenaltyWeightScheduler


def dynamic_partition(data, partitions, num_partitions):
    res = []
    for i in range(num_partitions):
        flags = (partitions == i)
        res.append(data[flags.nonzero().squeeze(1)])
    return res


class IRMv1Trainer(BaseTrainer):
    """
    Implementation for:
    Arjovsky, Martin, et al. "Invariant risk minimization." (ICLR 2021).

    This attempts to learn representations that enable the same classifier to be optimal across environments.
    For our implementation, we use the explicit groups based on the explicit bias variables including the class label to form the environments.
    We uniformly sample from environments within each batch (i.e., our implementation assumes balanced group sampling).
    """

    def __init__(self, option):
        super(IRMv1Trainer, self).__init__(option)
        self.group_names = None
        self.env_to_data_loader = None
        self.num_envs_per_batch = option.num_envs_per_batch
        self.max_groups = option.max_groups 
        self.penalty_weight_scheduler = PenaltyWeightScheduler(iter_to_max = self.option.penalty_weight_ascend_iter_n, 
                                                               init_val = 0, 
                                                               max_val = self.option.grad_penalty_weight)
        self.cur_train_iter = 0
        assert self.option.batch_size % self.num_envs_per_batch == 0, \
            "Batch size is not exactly divisible by the number of environments within that batch"

    # def compute_gradient_penalty(self, logits, y):
    def compute_gradient_penalty(self, logits, labels, envs, weight) -> torch.Tensor:
        """
        Gradient of the risk when all classifier weights are set to 1 (i.e., the IRMv1 regularizer)
        Based on unbiased estimation of IRMV1 (Sec 3.2) and Appendix D of https://arxiv.org/pdf/1907.02893.pdf
        It assumes that each batch contains per-environment samples from 'num_envs_per_batch' environments in an ordered manner.

        :param logits:
        :param y:
        :return:
        """
        dummy_classifier = torch.tensor(1.).cuda().requires_grad_()
        # loss = self.loss(logits * dummy_classifier, y.squeeze())
        # start_ix = 0
        # grad_loss = 0
        # for env in np.arange(0, self.num_envs_per_batch):
        #     end_ix = start_ix + self.option.batch_size // self.num_envs_per_batch
        #     env_loss = loss[start_ix:end_ix]
        #     loss1 = env_loss[:len(env_loss) // 2]
        #     loss2 = env_loss[len(env_loss) // 2:]
        #     grad1 = torch.autograd.grad(loss1.mean(), [dummy_classifier], create_graph=True)[0]
        #     grad2 = torch.autograd.grad(loss2.mean(), [dummy_classifier], create_graph=True)[0]
        #     grad_loss += grad1 * grad2
        #     start_ix = end_ix
        env_part_logits = dynamic_partition(logits, envs, self.option.num_classes)
        env_part_labels = dynamic_partition(labels, envs, self.option.num_classes)
        if weight is not None:
            env_part_weights = dynamic_partition(weight, envs, self.option.num_classes)
            penalty = torch.tensor(0.0).to(logits.device)
            count = 0.0
            for env_logit, env_label, env_weight in zip(env_part_logits, env_part_labels, env_part_weights):
                if len(env_logit) == 0: continue
                count += 1
                dummy_cost = torch.mean(self.loss(env_logit * dummy_classifier, env_label) * env_weight, dim=0)
                # test the group balanced penalty : N
                # dummy_cost = torch.mean(self.loss(env_logit * dummy_classifier, env_label) * env_weight, dim=0) / len(env_label) * len(labels)
                # test the group balanced penalty
                # dummy_cost = torch.mean(self.loss(env_logit * dummy_classifier, env_label) * env_weight, dim=0) / len(env_label) * len(labels)
                penalty += (torch.autograd.grad(dummy_cost, [dummy_classifier], create_graph=True)[0]**2).sum()
            # penalty /= self.option.num_classes
            # with the group balanced penalty
            # penalty /= count
            penalty /= count
        else:
            penalty = torch.tensor(0.0).to(logits.device)
            count = 0.0
            for env_logit, env_label in zip(env_part_logits, env_part_labels):
                if len(env_logit) == 0: continue
                count += 1
                dummy_cost = torch.mean(self.loss(env_logit * dummy_classifier, env_label), dim=0)
                penalty += (torch.autograd.grad(dummy_cost, [dummy_classifier], create_graph=True)[0]**2).sum()
            # penalty /= self.option.num_classes
            penalty /= count

        return penalty

    def compute_weighted_loss(self, losses, y, envs, weights):
        if weights is not None:
            all_loss = losses * weights
            # return all_loss.mean()
            env_part_losses = dynamic_partition(all_loss, envs, self.max_groups)
            final_loss = 0.0
            env_count = 0.0
            for env_loss in env_part_losses:
                if len(env_loss) == 0: continue
                final_loss += env_loss.mean()
                env_count += 1
            return final_loss / env_count
        else:
            return losses.mean()

    def train(self, train_loader, test_loaders=None, unbalanced_train_loader=None):
        self.before_train(train_loader, test_loaders)
        start_epoch = 1
        print("self.option.env_balance_sampler: ", self.option.env_balance_sampler)
        if self.option.env_balance_sampler:
            orig_loader = train_loader
            batch_sampler = EnvironmentWiseBatchSampler(self.option.batch_size, orig_loader, self.num_envs_per_batch)
            dataset = orig_loader.dataset
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
            train_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                    num_workers=orig_loader.num_workers, collate_fn=orig_loader.collate_fn)

        for epoch in range(start_epoch, self.option.epochs + 1):
            self._train_epoch(epoch, train_loader)
            self._after_one_epoch(epoch, test_loaders)
        self.after_all_epochs()

    def _train_epoch(self, epoch, data_loader):
        self._mode_setting(is_train=True)
        for batch_ix, batch in enumerate(data_loader):
            batch = self.prepare_batch(batch)
            out = self.forward_model(self.model, batch)
            logits = out['logits']
            # Unbiased IRMv1 goes through each environment before doing a backward pass
            # However this is not scalable e.g., when # of environments are in 100s or 1000s,
            # so we randomly sample certain environments in every batch
            batch_losses = self.loss(logits, torch.squeeze(batch['y']))
            batch_loss = self.compute_weighted_loss(batch_losses, batch['y'], batch['group_ix'], batch['weight'])
            penalty_weight = self.penalty_weight_scheduler.step(self.cur_train_iter)
            grad_penalty = penalty_weight * self.compute_gradient_penalty(logits, batch['y'], batch['group_ix'], batch['weight'])
            self.loss_visualizer.update(f'Train', 'Main Loss', batch_losses.mean().item())
            self.loss_visualizer.update(f'Train', 'Grad Penalty', grad_penalty.mean().item())

            self.optim.zero_grad()
            # loss = batch_losses.mean() + grad_penalty.mean()
            loss = batch_loss + grad_penalty
            weight_norm = torch.tensor(0.).cuda()
            for w in self.model.parameters():
                weight_norm += w.norm().pow(2)
            loss += self.option.l2_reg_weight * weight_norm
            if penalty_weight > 1.0: # the same as in IRM
                loss /= penalty_weight
            
            loss.backward(retain_graph=True)  # Cannot go through each environment before calling backward()
            if self.option.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.option.grad_clip)
            self.optim.step()
            self.cur_train_iter += 1

        self.optim.zero_grad()
        self._after_train_epoch(epoch)


class EnvironmentWiseBatchSampler():
    def __init__(self, batch_size, data_loader, num_envs_per_batch):
        """
        We first identify the environment for each data item
        For each batch, we randomly sample from random environments
        """
        self.num_items = 0
        self.env_to_dataset_ixs = {}
        self.batch_size = batch_size
        self.num_envs_per_batch = num_envs_per_batch

        # Do one pass through the train_loader to get indices per env
        for batch_ix, batch in enumerate(data_loader):
            for dix, gix in zip(batch['dataset_ix'], batch['group_ix']):
                if gix not in self.env_to_dataset_ixs:
                    self.env_to_dataset_ixs[gix] = []
                self.env_to_dataset_ixs[gix].append(dix)
                self.num_items += 1
        self.env_keys = list(self.env_to_dataset_ixs.keys())
        # logging.getLogger().info(f"env keys {self.env_keys}")
        # for gix in self.env_to_dataset_ixs:
        #     logging.getLogger().info(f"Env Key: {gix} Cnt: {len(self.env_to_dataset_ixs[gix])}")

    def __iter__(self):
        num_batches_per_epoch = self.__len__()
        curr_batch_cnt = 0

        while curr_batch_cnt <= num_batches_per_epoch:
            # Randomly select some environments per batch
            env_ixs = np.random.choice(self.env_keys, self.num_envs_per_batch, replace=False)  # Randomly sample some environments

            # Randomly select within each of the chosen environments
            batch = []

            for env_ix in env_ixs:
                for b in np.arange(self.batch_size // self.num_envs_per_batch):
                    dix = random.choice(self.env_to_dataset_ixs[env_ix])
                    batch.append(dix)

            curr_batch_cnt += 1
            yield batch

    def __len__(self):
        # The total budget per epoch is self.num_items
        return self.num_items // self.batch_size
