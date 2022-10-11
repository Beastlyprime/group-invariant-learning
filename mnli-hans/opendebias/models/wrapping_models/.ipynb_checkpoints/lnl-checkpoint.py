from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.autograd import grad, grad_mode
from torch.nn.modules import transformer
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
from allennlp.modules import FeedForward
from opendebias.modules.losses import EBDLoss
from overrides import overrides
from copy import deepcopy
import numpy as np
from torch import nn
import inspect
from opendebias.models.utils import dynamic_partition
from torch.autograd import Function


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None
    
revgrad = RevGrad.apply

class RevGrad(torch.nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)



@Model.register('LNL')
class LearningNotToLearn(Model):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        entropy_loss_weight: float,
        grad_reverse_factor: float,
        bias_predictor_output_dim: int,
        bias_predictor_input_dim: int = 768,
        bias_predictor_hidden_dim: int = 0,
        bias_predictor_dropout: float = 0,
        bias_predictor_activation: str = "relu",
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._model = model
        self._loss = nn.CrossEntropyLoss()
        self._entropy_loss_weight = entropy_loss_weight
        self._grad_reverse_factor = grad_reverse_factor
        self._metrics = {} if metrics is None else metrics
        self._label_namespace = label_namespace
        self._label_count = vocab.get_vocab_size(self._label_namespace)
        self._namespace = namespace

        self.entropy = Average()
        self.non_entropy_loss = Average()
        self.bias_loss = Average()
        self._rev = RevGrad(self._grad_reverse_factor)

        if bias_predictor_activation.lower() == 'relu':
            Activation = nn.ReLU
        elif bias_predictor_activation.lower() == 'tanh':
            Activation = nn.Tanh
        else:
            raise NotImplementedError()

        if bias_predictor_hidden_dim == 0:
            self._bias_predictor = nn.Linear(bias_predictor_input_dim, bias_predictor_output_dim)
        else:
            self._bias_predictor = nn.Sequential(
                nn.Linear(bias_predictor_input_dim, bias_predictor_hidden_dim),
                Activation(),
                nn.Dropout(bias_predictor_dropout),
                nn.Linear(bias_predictor_hidden_dim, bias_predictor_output_dim),
            )
        initializer(self)

    @overrides
    def forward(
        self, 
        **kw_input,
    ) -> Dict[str, torch.Tensor]:
        model_output = self._model(**{k:v for k, v in kw_input.items() if k in inspect.getfullargspec(self._model.forward).args and k != 'label'})
        output_dict: Dict[str, Any] = {}
        output_dict['logits'] = model_output['logits']
        output_dict['probs'] = torch.softmax(model_output['logits'], dim=-1)

        if 'label' in kw_input:
            logits = output_dict['logits']
            labels = kw_input['label']
            hidden = model_output['hidden']
            bias = kw_input.get('bias', None)
            model_loss = self._loss(logits, labels)
            if self.training and bias is None: raise NotImplementedError()
            if bias is not None:
                bias_logits = self._bias_predictor(hidden)
                bias_softmax = torch.nn.functional.softmax(bias_logits, dim=1) + 1e-8
                bias_entropy_loss = torch.mean(
                    torch.sum(bias_softmax * torch.log(bias_softmax), 1))
                reverse_hidden = self._rev(hidden)
                bias_logits = self._bias_predictor(reverse_hidden)
                adv_bias_loss = self._loss(bias_logits, bias)
                loss = model_loss + bias_entropy_loss * self._entropy_loss_weight + adv_bias_loss

                self.entropy(bias_entropy_loss.detach())
                self.bias_loss(adv_bias_loss.detach())
            else:
                loss = model_loss
                
            self.non_entropy_loss(model_loss.detach())
        
            for key, metric in self._metrics.items():
                metric(logits, labels)
            output_dict['loss'] = loss
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for key, metric in self._metrics.items():
            metrics[f'{key}'] = metric.get_metric(reset)
        metrics['entropy'] = self.entropy.get_metric(reset)
        metrics['ce_loss'] = self.non_entropy_loss.get_metric(reset)
        metrics['bias_loss']=self.bias_loss.get_metric(reset)

        return metrics
    
    def bias_only_model_load_weight(self):
        return