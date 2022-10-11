from enum import unique
from os import P_NOWAIT
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.autograd import grad, grad_mode
from allennlp.common import Registrable
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models import model
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder)
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric, Average, metric
from opendebias.modules.losses import EBDLoss
from overrides import overrides
from copy import deepcopy
import numpy as np
from torch import nn
import inspect
from opendebias.models.utils import dynamic_partition
from opendebias.models.wrapping_models.group_models.base import GroupBase

@Model.register('rex')
class REx(GroupBase):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        weights_file_path: str, 
        weight_adapt: float = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, model, weights_file_path, weight_adapt, label_namespace, namespace, initializer, metrics, **kwargs)


    @overrides
    def calculate_penalty(self, all_loss, logits, labels, envs, unique_env_count, weight=None, hidden=None) -> torch.Tensor:
        if weight is not None:
            all_loss = all_loss * weight
        env_part_losses = dynamic_partition(all_loss, envs, unique_env_count)
        # env_part_losses = [torch.mean(env_part_loss) for env_part_loss in ] # first reduce it
        env_mean_loss = 0.0
        for env_loss in env_part_losses:
            if len(env_loss) == 0: continue
            env_mean_loss += env_loss.mean()
        env_mean_loss /= unique_env_count

        penalty = 0.0
        for env_loss in env_part_losses:
            if len(env_loss) == 0: continue
            penalty += (env_loss.mean() - env_mean_loss) ** 2
        penalty /= unique_env_count
        return penalty

    @overrides
    def calculate_loss(self, all_loss, logits, labels, envs, unique_env_count, weight, hidden=None) -> torch.Tensor:
        if weight is not None:
            all_loss = all_loss * weight
        env_part_losses = dynamic_partition(all_loss, envs, unique_env_count)
        final_loss = 0.0
        for env_loss in env_part_losses:
            if len(env_loss) == 0: continue
            final_loss += env_loss.mean()
        return final_loss / unique_env_count

    @overrides
    def rescale_loss(self, loss) -> torch.Tensor:
        if self._weight_adapt > 1.0:
            return loss / self._weight_adapt
        else:
            return loss
    

@Model.register('cRex')
class ConditionalRex(REx):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        weight_adapt: float = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, model, weight_adapt, label_namespace, namespace, initializer, metrics, **kwargs)

    
    @overrides
    def calculate_penalty(self, all_loss, logits, labels, envs, unique_env_count, weight, hidden=None) -> torch.Tensor:
        class_part_all_loss = dynamic_partition(all_loss, labels, self._label_count)
        class_part_envs = dynamic_partition(envs, labels, self._label_count)
        penalty = 0.0
        label_count = 0.0
        for class_loss, class_envs in zip(class_part_all_loss, class_part_envs):
            if len(class_loss) == 0: continue
            penalty += super().calculate_penalty(None, None, None, class_envs, unique_env_count, None)
            label_count += 1
        penalty /= label_count
        return penalty