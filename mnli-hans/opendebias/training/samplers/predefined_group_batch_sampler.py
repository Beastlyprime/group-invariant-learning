from numpy.core.defchararray import index
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler, BatchSampler
import logging
from typing import List, Iterable, Tuple, Dict, Optional, overload
import random
import math
import os
import json
import numpy as np 
import torch

from torch.utils import data

from allennlp.common.util import lazy_groups_of
from overrides import overrides
from allennlp.data.fields import ListField, MetadataField, LabelField

logger = logging.getLogger(__name__)

@BatchSampler.register("predefined_group")
class PredefinedGroupSampler(BucketBatchSampler):
    def __init__(
        self,
        data_source: data.Dataset,
        predefined_group_file: List[Dict[str, str]],
        batch_size: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
        num_group_per_batch: Optional[int] = None,
    ):
        super().__init__(data_source, batch_size, sorting_keys, padding_noise, drop_last)
        self.dataset_name = data_source[0]['metadata']['dataset_name']
        self._num_group_per_batch = num_group_per_batch
        self.group2index, self.index2group = self._read_predefined_group_file(predefined_group_file)
        self.group_count = len(self.group2index[self.dataset_name].keys())
        self.all_groups = list(self.group2index[self.dataset_name].keys())

        # add instance group idx
        for instance in data_source:
            idx = instance["index"]._label_id
            instance.add_field('group', LabelField(self.index2group[self.dataset_name][idx], skip_indexing=True))
            instance.add_field('group_count', LabelField(self._group_count, skip_indexing=True))

        # for idx, gid in self.index2group[self.dataset_name].items():
            # if idx < len(data_source):
                # instance = data_source[idx]
                # instance.add_field('group', LabelField(self.index2group[self.dataset_name][idx], skip_indexing=True))
                # instance.add_field('group_count', LabelField(self._group_count, skip_indexing=True))
                

    def _read_predefined_group_file(self, args: List[Dict[str, str]]):
        group2index = {}
        index2group = {}
        max_gid = 0
        for arg in args:
            if len(arg) == 0:
                continue
            if arg['dataset_name'] != self.dataset_name:
                continue
            if not os.path.exists(arg['file_name']):
                logger.warning('{} is not exists'.format(arg['file_name']))
                group2index[arg['dataset_name']] = None
                index2group[arg['dataset_name']] = None
            else:
                with open(arg['file_name'], 'r') as f:
                    group_dict = json.load(f)
                group2index[arg['dataset_name']] = {}
                index2group[arg['dataset_name']] = {}
                for idx, gid in enumerate(group_dict['group']):
                    group2index[arg['dataset_name']][gid] = idx 
                    index2group[arg['dataset_name']][idx] = gid
                    max_gid = max(max_gid, gid)
        self._group_count = max_gid + 1
        return group2index, index2group

    @overrides
    def __iter__(self) -> Iterable[List[int]]:
        if self._num_group_per_batch is None:
            yield from super().__iter__()
        else:
            indices, _ = self._argsort_by_padding(self.data_source)
            batches = []
            for batch_indices in lazy_groups_of(indices, self.batch_size):
                batch_indices = list(batch_indices)
                cur_batch_size = len(list(batch_indices))
                batch_indices = []  # renew batch_indices
                if self.drop_last and cur_batch_size < self.batch_size:
                    continue
                select_gids = random.choices(self.all_groups, self._num_group_per_batch)
                per_group_count = cur_batch_size // self._num_group_per_batch
                for gid in select_gids:
                    batch_indices.extend(random.choices(self.group2index[self.dataset_name][gid], per_group_count))
                if len(batch_indices) < cur_batch_size:
                    batch_indices.extend(random.choices(self.group2index[self.dataset_name][gid], cur_batch_size-len(batch_indices)))
                batches.append(batch_indices)
            random.shuffle(batches)
            for batch in batches:
                yield batch