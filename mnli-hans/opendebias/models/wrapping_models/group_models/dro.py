from enum import unique
from opendebias.models.wrapping_models import group_models
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import final

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
import math
import inspect
from opendebias.models.utils import dynamic_partition
from opendebias.models.wrapping_models.group_models.base import GroupBase

@Model.register('dro')
class GroupDRO(GroupBase):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        weights_file_path: str, 
        weight_adapt: float,  # here weight adapt is learning rate for dro weights
        # Cs: List[float],   
        Cs: float,   
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, model, weights_file_path, weight_adapt, label_namespace, namespace, initializer, metrics, **kwargs)
        # self._Cs = torch.tensor(Cs).float()
        self._Cs = Cs
        self._dro_weights = None
       
    
    @overrides
    def calculate_penalty(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        penalty = torch.tensor(0.0).to(logits.device)
        return penalty
    
    @overrides
    def calculate_loss(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        if not self.training:
            return torch.mean(all_loss)
        if self._dro_weights is None:
            # self._dro_weights = torch.ones(unique_env_count).float()
            self._dro_weights = self._dro_weights / torch.sum(self._dro_weights, dim=-1, keepdim=True)
            self._dro_weights = self._dro_weights.to(logits.device)
            self._dro_weights = nn.Parameter(self._dro_weights.to(logits.device))
        
        env_part_losses = dynamic_partition(all_loss.detach(), envs, unique_env_count)
        group_adujst_env_part_losses = []
        for eidx in range(len(env_part_losses)):
            assert len(env_part_losses[eidx]) != 0
            # group_adujst_env_part_losses.append(torch.mean(env_part_losses[eidx] + self._Cs[eidx] / torch.sqrt(len(env_part_losses))))
            group_adujst_env_part_losses.append(torch.mean(env_part_losses[eidx] + self._Cs / math.sqrt(len(env_part_losses))))
        group_adujst_env_part_losses = torch.stack(group_adujst_env_part_losses, dim = 0)
        # Normalising here seemed to help in early experiments overall
        group_adujst_env_part_losses = group_adujst_env_part_losses / torch.sum(group_adujst_env_part_losses)

        # update dro weights
        self._dro_weights = self._dro_weights * torch.exp(self._weight_adapt * group_adujst_env_part_losses.detach())
        self._dro_weights = self._dro_weights / torch.sum(self._dro_weights)

        # calucate loss
        final_loss = 0.0
        final_loss = torch.sum(self._dro_weights * torch.stack([torch.mean(env_loss) for env_loss in env_part_losses], dim=0))
        return final_loss
    

@Model.register('cdro')
class ConditionalGroupDRO(GroupDRO):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        weight_adapt: float,
        Cs: List[float],
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, model, weight_adapt, Cs, label_namespace, namespace, initializer, metrics, **kwargs)

    
    @overrides
    def calculate_loss(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        if not self.training:
            return torch.mean(all_loss)

        if self._dro_weights is None:
            weights = torch.tensor([[1.0 for __ in range(unique_env_count)] for _ in range(self._label_count)]).float().to(logits.device)
            weights = weights / torch.sum(weights, dim=-1, keepdim=True)
            self._dro_weights = weights

        class_part_losses = dynamic_partition(all_loss, labels, self._label_count)
        class_part_envs = dynamic_partition(envs, labels, self._label_count)
        final_loss = 0.0
        for lidx, (class_losses, class_envs) in enumerate(zip(class_part_losses, class_part_envs)):
            if len(class_losses) == 0: continue
            env_part_class_losses = torch.stack([torch.mean(eloss) for eloss in dynamic_partition(class_losses, class_envs, unique_env_count)], dim=0)
            final_loss += torch.sum(env_part_class_losses * self._dro_weights[lidx])
        final_loss /= self._label_count
        return final_loss