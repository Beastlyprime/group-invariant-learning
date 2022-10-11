import logging
from typing import List, Iterable, Tuple, Dict, Optional
import random
import math
import os
import json
import numpy as np 
import torch

from torch.utils import data

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers import BatchSampler
from copy import deepcopy
from allennlp.data.fields import ListField, MetadataField

logger = logging.getLogger(__name__)

def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise

class BiasAwareBatchSamplerBase(BatchSampler):
    def __init__(
        self,
        data_source: data.Dataset,
        batch_size: int,
        bias_prediction_file: List[Dict[str, str]],
        K: int = None,
        K_pos: int = None,
        K_neg: int = None,
        stratified_sample: bool = True,
        padding_noise: Optional[float] = 0.1,
        sorting_keys: List[str] = None,
        label_namespace: str = "labels",
    ):
        self.instances = data_source
        self.vocab = data_source.vocab
        self._label_namespace = label_namespace
        self.batch_size = batch_size
        self.data_source = data_source
        self.dataset_name = data_source[0]['metadata']['dataset_name']
        self.drop_last = False
        self.K = K
        self.K_pos = K_pos
        self.K_neg = K_neg
        assert not ((self.K_pos is not None) ^ (self.K_neg is not None))
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        bias_logits = self._read_prediction_file(bias_prediction_file)
        for instance in self.data_source:
            instance.index_fields(self.vocab)
        self.groupped_instances = self.group_instances(self.data_source, bias_logits)    # group_count x label_count
        self.stratified_sample = stratified_sample
        group_count, label_count = len(self.groupped_instances), self.vocab.get_vocab_size(self._label_namespace)
        self.group_count = group_count
        self.label_count = label_count

        if self.stratified_sample:
            # for random select another group or class uniformly
            self.another_group_chooses = [[j for j in range(group_count) if j != i] for i in range(group_count)]
            self.another_label_chooses = [[j for j in range(label_count) if j != i] for i in range(label_count)]
        else:
            # create cross_group and cross_label list
            self.cross_groupped_instances = [[[]  for __ in range(group_count)] for _ in range(label_count)]    # label_count x group_count      same label, different group instances
            self.cross_label_instances = [[[] for __ in range(label_count)] for _ in range(group_count)]        # group_count x label_count      same group, different label instances

            for label in range(label_count):
                for group in range(group_count):
                    for gidx in range(group_count):
                        if gidx == group: continue
                        self.cross_groupped_instances[label][group] += self.groupped_instances[gidx][label]
            for group in range(group_count):
                for label in range(label_count):
                    for lidx in range(label_count):
                        if lidx == label: continue
                        self.cross_label_instances[group][label] += self.groupped_instances[group][lidx]

    def sample(self, target_group, target_y, num, cur_idx = None):
        raise NotImplementedError()

    def sample_instances(self, instance_idx):
        raise NotImplementedError()


    def another_group(self, group_idx, y):
        return random.choice(self.another_group_chooses[group_idx])

    def another_class(self, group_idx, y):
        return random.choice(self.another_label_chooses[y])

    def group_instances(self, instances, bias_logits):
        dataset = instances[0]['metadata']['dataset_name']
        bias = bias_logits[dataset]
        label_count = self.vocab.get_vocab_size(self._label_namespace)
        group = set([tuple(lp.cpu().numpy().tolist()) for lp in bias])
        lp2gid = {lp:idx for idx, lp in enumerate(sorted(group))}
        instance_group = [lp2gid[tuple(lp.cpu().numpy().tolist())] for lp in bias]        # [gid1, gid2, ..., gidK]
        groupped_instances = [[[] for __ in range(label_count)] for _ in range(len(lp2gid))]   # shape group * y * x   every x is a instance_index
        for idx, instance in enumerate(instances):
            index = instance["index"].label
            # index = idx
            y = instance['label']._label_id
            gid = instance_group[index]
            groupped_instances[gid][y].append(index)
        self.instance_group = instance_group
        return groupped_instances

    def merge_instance(self, field_name, instance, to_merge_instance_idxes):
        if len(to_merge_instance_idxes) > 0:
            list_field = ListField([self.instances[idx]['tokens'] for idx in to_merge_instance_idxes])
            label_list_field = ListField([self.instances[idx]['label'] for idx in to_merge_instance_idxes])
            instance.add_field(field_name, list_field, vocab=self.vocab)
            instance.add_field('{}_label'.format(field_name), label_list_field, vocab=self.vocab)
            instance.add_field('{}_index'.format(field_name), MetadataField({"id": to_merge_instance_idxes}))
        return instance
    
    def _read_prediction_file(self, args: List[Dict[str, str]]):
        bias_logits: Dict[str, np.ndarray] = {}
        args = [] if args is None else args
        for arg in args:
            if len(arg) == 0:
                continue
            if not os.path.exists(arg['file_name']):
                logger.warning('{} is not exists'.format(arg['file_name']))
                bias_logits[arg['dataset_name']] = None
            else:
                logits = self._read(arg['file_name'], arg['dataset_name'])
                bias_logits[arg['dataset_name']] = torch.from_numpy(logits).float()
        return bias_logits

    def _read(self, fn, dataset_name):
        with open(fn, 'r') as f:
            bias = json.load(f)
        bias_logits_list = [None] * len(bias['logits'])
        dim = len(bias['logits'][0])
        index2token = bias['index2token']
        for line_id, v in zip(bias['index'], bias['logits']):
            bias_logits = [None for _ in range(dim)]
            for i in range(dim):
                label = index2token[str(i)]
                index = self.vocab.get_token_index(label, self._label_namespace)
                bias_logits[index] = v[i]
            bias_logits_list[int(line_id)] = np.array(bias_logits)
        bias_logits_list = np.array(bias_logits_list)
        return bias_logits_list

    def __iter__(self) -> Iterable[List[int]]:
        # reshuffle all groupped_instances
        indices, _ = self._argsort_by_padding(self.data_source)
        batches = []    
        for group in lazy_groups_of(indices, self.batch_size):
            batch_indices = list(group)
            for idx in batch_indices:
                self.sample_instances(idx)  # sample instance
            batches.append(batch_indices)        
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        batch_count_float = len(self.data_source) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

    def _guess_sorting_keys(self, instances: Iterable[Instance], num_instances: int = 10) -> None:
        """
        Use `num_instances` instances from the dataset to infer the keys used
        for sorting the dataset for bucketing.

        # Parameters

        instances : `Iterable[Instance]`, required.
            The dataset to guess sorting keys for.
        num_instances : `int`, optional (default = `10`)
            The number of instances to use to guess sorting keys. Typically
            the default value is completely sufficient, but if your instances
            are not homogeneous, you might need more.
        """
        max_length = 0.0
        longest_field: str = None
        for i, instance in enumerate(instances):
            instance.index_fields(self.vocab)
            for field_name, field in instance.fields.items():
                length = len(field)
                if length > max_length:
                    max_length = length
                    longest_field = field_name
            if i > num_instances:
                # Only use num_instances instances to guess the sorting keys.
                break

        if not longest_field:
            # This shouldn't ever happen (you basically have to have an empty instance list), but
            # just in case...
            raise AssertionError(
                "Found no field that needed padding; we are surprised you got this error, please "
                "open an issue on github"
            )
        self.sorting_keys = [longest_field]

    def _argsort_by_padding(
        self, instances: Iterable[Instance]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        if not self.sorting_keys:
            logger.info("No sorting keys given; trying to guess a good one")
            self._guess_sorting_keys(instances)
            logger.info(f"Using {self.sorting_keys} as the sorting keys")
        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:
                if field_name not in instance.fields:
                    raise ConfigurationError(
                        f'Sorting key "{field_name}" is not a field in instance. '
                        f"Available fields/keys are {list(instance.fields.keys())}."
                    )
                lengths.append(len(instance.fields[field_name]))

                noisy_lengths.append(add_noise_to_value(lengths[-1], self.padding_noise))
            instances_with_lengths.append((noisy_lengths, lengths, instance))
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return (
            [instance_with_index[-1] for instance_with_index in with_indices],
            [instance_with_index[0][1] for instance_with_index in with_indices],
        )