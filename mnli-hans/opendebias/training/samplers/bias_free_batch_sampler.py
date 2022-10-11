import json
import logging
import math
import os
import random
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.fields import ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.samplers import BatchSampler
from opendebias.training.samplers.bias_aware_batch_sampler_base import \
    BiasAwareBatchSamplerBase
from overrides import overrides
from torch.utils import data

logger = logging.getLogger(__name__)

@BatchSampler.register("bias-free")
class BiasFreeSampler(BiasAwareBatchSamplerBase):
    def __init__(
        self,
        data_source: data.Dataset,
        batch_size: int,
        K: int = None,
        K_pos: int = None,
        K_neg: int = None,
        stratified_sample: bool = True,
        bias_prediction_file = None,
        label_namespace: str = "labels",
        padding_noise: Optional[float] = 0.1,
        sorting_keys: List[str] = None
    ):
        super().__init__(data_source, batch_size, None, K, K_pos, K_neg, stratified_sample, padding_noise, sorting_keys, label_namespace)


    @overrides
    def group_instances(self, instances, bias_logits):
        label_count = self.vocab.get_vocab_size(self._label_namespace)
        bias = torch.from_numpy(np.zeros((len(instances), label_count))).float()
        return super().group_instances(
            instances,
            {instances[0]['metadata']['dataset_name']:bias}
        )

    @overrides
    def sample(self, target_group, target_y, num, cur_idx = None):
        target_instance_idxes = self.groupped_instances[target_group][target_y]
        res_idxes = []
        while len(res_idxes) < num:
            sample_idx = random.sample(target_instance_idxes, k=1)[0]
            if sample_idx == cur_idx:
                continue
            res_idxes.append(sample_idx)
        return res_idxes

    @overrides
    def sample_instances(self, instance_idx):
        instance = self.instances[instance_idx]
        group_idx = self.instance_group[instance_idx]
        assert group_idx == 0
        y = instance['label']._label_id

        # sample pos
        pos_idxes, neg_idxes = [], []
        pos_count = self.K_pos if self.K_pos is not None else self.K
        neg_count = self.K_neg if self.K_neg is not None else self.K

        if self.stratified_sample:
            for _ in range(pos_count):
                pos_idxes.extend(self.sample(group_idx, y, 1, cur_idx=instance_idx))  #  same group, same class
            for _ in range(neg_count):
                neg_idxes.extend(self.sample(group_idx, self.another_class(group_idx, y), 1, cur_idx=instance_idx))    # same group, different class
        else:
            raise NotImplementedError()
            pos_idxes = random.sample(self.cross_groupped_instances[y][group_idx], pos_count)
            neg_idxes = random.sample(self.cross_label_instances[group_idx][y], neg_count)

        
        # merge pos and neg into instances
        self.merge_instance('positives', instance, pos_idxes)
        self.merge_instance('negatives', instance, neg_idxes)
        return pos_idxes, neg_idxes
