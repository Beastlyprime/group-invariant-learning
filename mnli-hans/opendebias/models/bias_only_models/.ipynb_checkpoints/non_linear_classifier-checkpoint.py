import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator
from opendebias.models.loadable_model import LoadableModel
from overrides import overrides

logger = logging.getLogger(__name__)


@Model.register("non-linear")
class NonLinearClassifier(LoadableModel):
    def __init__(
        self,
        vocab: Vocabulary,
        feedforward: FeedForward,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._label_namespace = label_namespace
        self._feedforward = feedforward
        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._feedforward.get_output_dim(), self._num_labels)
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)
    
    def forward(self, hidden, label=None):
        transformed_hidden = self._feedforward(hidden)
        logits = self._classification_layer(transformed_hidden)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
        return output_dict
