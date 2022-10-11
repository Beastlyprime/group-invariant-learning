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

class GroupBase(Model):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        weights_file_path: str = None, 
        weight_adapt: float = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._model = model
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self._weight_adapt = weight_adapt
        self._metrics = {} if metrics is None else metrics
        self._label_namespace = label_namespace
        self._label_count = vocab.get_vocab_size(self._label_namespace)
        self._namespace = namespace
        self.penalty = Average()
        self.non_penalty_loss = Average()
        self._weights = torch.Tensor(np.load(weights_file_path)) if weights_file_path is not None else None

        initializer(self)

    
    def calculate_loss(self, all_loss, logits, labels, envs, unique_env_count, weights, hidden=None) -> torch.Tensor:
        raise NotImplementedError()

    def calculate_penalty(self, all_loss, logits, labels, envs, unique_env_count, hidden=None) -> torch.Tensor:
        raise NotImplementedError()
        
    def rescale_loss(self, loss) -> torch.Tensor:
        return loss

    @overrides
    def forward(
        self, 
        **kw_input,
    ) -> Dict[str, torch.Tensor]:
        model_output = self._model(**{k:v for k, v in kw_input.items() if k in inspect.getfullargspec(self._model.forward).args and k != 'label'})
        output_dict: Dict[str, Any] = {}
        output_dict['logits'] = model_output['logits']
        output_dict['probs'] = torch.softmax(model_output['logits'], dim=-1)

        cuda0 = torch.device('cuda')
        index = kw_input['index']
        if self._weights is not None:
            batch_weights = self._weights[np.array(index.cpu())].to(cuda0)
        else:
            batch_weights = torch.ones(len(index)).to(cuda0)

        if 'label' in kw_input:
            logits = output_dict['logits']
            labels = kw_input['label']
            hidden = model_output['hidden']
            envs = kw_input.get('group', None)

            all_loss = self._loss(logits, labels)
            if self.training and envs is None: raise NotImplementedError()
            if envs is not None:
                env_count = kw_input['group_count'][0].cpu().item()
                penalty = self.calculate_penalty(all_loss, logits, labels, envs, env_count, batch_weights, hidden)
                cost = self.calculate_loss(all_loss, logits, labels, envs, env_count, batch_weights, hidden)
                loss = cost + self._weight_adapt * penalty   # final cost
                self.rescale_loss(loss)
                self.penalty(penalty.detach().item())
                self.non_penalty_loss(cost.detach().item())
            else:
                loss = torch.mean(all_loss)
            for key, metric in self._metrics.items():
                metric(logits, labels, weights=batch_weights)
            output_dict['loss'] = loss
        return output_dict

    # for output metrics
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for key, metric in self._metrics.items():
            metrics[f'{key}'] = metric.get_metric(reset)
        metrics['penalty'] = self.penalty.get_metric(reset)
        metrics['ce_loss'] = self.non_penalty_loss.get_metric(reset)
        return metrics
    
    def bias_only_model_load_weight(self):
        return