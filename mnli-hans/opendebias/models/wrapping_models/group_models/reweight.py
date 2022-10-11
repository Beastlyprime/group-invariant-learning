from enum import unique
from re import S
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
from allennlp.training.metrics import Metric, Average
from opendebias.modules.losses import EBDLoss
from overrides import overrides
from copy import deepcopy
import numpy as np
from torch import nn
import inspect
from opendebias.models.utils import dynamic_partition
from opendebias.models.wrapping_models.group_models.base import GroupBase
import torch.nn.functional as F


@Model.register('reweight')
class Reweight(GroupBase):
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
        reverse: bool = False,
        **kwargs,
    ) -> None:
        self._reverse = reverse
        super().__init__(vocab, model, weights_file_path, weight_adapt, label_namespace, namespace, initializer, metrics, **kwargs)

    
    @overrides
    def calculate_penalty(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        return 0.0
    
    @overrides
    def calculate_loss(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        weight_adapt = self._weight_adapt
        loss_weights = envs.float() + (1./(1.0+weight_adapt))*(1-envs.float())
        loss_weights = loss_weights/torch.sum(loss_weights)
        final_loss = torch.mean(loss_weights * all_loss)
        return final_loss