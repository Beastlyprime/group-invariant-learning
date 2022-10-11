from typing import Any, Dict, List, Optional, Tuple

import torch
from allennlp.common import Registrable
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2SeqEncoder, Seq2VecEncoder,
                              TextFieldEmbedder)
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric
from opendebias.modules.losses import EBDLoss
from overrides import overrides
from copy import deepcopy


class EBDModelBase(Model):
    def __init__(
        self, 
        vocab: Vocabulary,
        bias_only_model: Model,
        main_model: Model,
        ebd_loss: EBDLoss,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        metrics: Optional[Dict[str, Metric]] = None,
        ret_hidden = False,
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._main_model = main_model
        self._bias_only_model = bias_only_model
        self._loss = ebd_loss

        self._main_metrics = {} if metrics is None else deepcopy(metrics)
        self._bias_only_metrics = {} if metrics is None else deepcopy(metrics)
        self._ensemble_metrics = {} if metrics is None else deepcopy(metrics)

        self._label_namespace = label_namespace
        self._namespace = namespace
        self._ret_hidden = ret_hidden
        initializer(self)


    def interaction(self,
                    **kw_input) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError()

    def forward_main_model(
        self,
        **kw_input,
    ) -> Dict[str, torch.Tensor]:
        main_model_output = self._main_model(**kw_input)
        return main_model_output
        

    @overrides
    def forward(
        self, 
        **kw_input,
    ) -> Dict[str, torch.Tensor]:
        main_model_output, bias_only_model_output = self.interaction(**kw_input)
        ensemble_output = self._loss(main_model_output, bias_only_model_output, kw_input.get("label", None))
        output_dict: Dict[str, Any] = {}
        # save output
        to_save_fields = ["probs", "logits", "loss"]
        for field in to_save_fields:
            if field in main_model_output:
                output_dict[f"main_{field}"] = main_model_output[field]

            if field in bias_only_model_output:
                output_dict[f"bias_only_{field}"] = bias_only_model_output[field]

            if field in ensemble_output and field != 'loss':
                output_dict[f"ensemble_{field}"] = ensemble_output[field]

        if self._ret_hidden:
            output_dict[f"hidden"] = main_model_output["hidden"]
        
        if "loss" in ensemble_output:
            output_dict["loss"] = ensemble_output["loss"]
        if "label" in kw_input:
            for metric in self._main_metrics.values():
                metric(main_model_output["logits"], kw_input["label"])
            for metric in self._bias_only_metrics.values():
                if "logits" in bias_only_model_output:
                    metric(bias_only_model_output["logits"], kw_input["label"])
            for metric in self._ensemble_metrics.values():
                if "logits" in ensemble_output:
                    metric(ensemble_output["logits"], kw_input["label"])
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for key, metric in self._main_metrics.items():
            metrics[f'{key}'] = metric.get_metric(reset)
        for key, metric in self._bias_only_metrics.items():
            metrics[f'bias_only_{key}'] = metric.get_metric(reset)
        for key, metric in self._ensemble_metrics.items():
            metrics[f'ensemble_{key}'] = metric.get_metric(reset)
        return metrics
    
    def bias_only_model_load_weight(self):
        self._bias_only_model.load_weight()
