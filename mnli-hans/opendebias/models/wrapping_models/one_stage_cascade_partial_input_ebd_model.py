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
from opendebias.models.wrapping_models.base_model import EBDModelBase
from overrides import overrides
import inspect


@Model.register("one_stage_cascade_partial_input")
class OneStageCascadePartialInputEBDModel(EBDModelBase):

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
        bias_only_loss_lambda: Optional[float] = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab, bias_only_model, main_model, ebd_loss,
            label_namespace, namespace, initializer, metrics
        )
        self._lambda = bias_only_loss_lambda
        

    @overrides
    def interaction(self,
                    **kw_input) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        keys = inspect.getfullargspec(self._main_model.forward).args
        main_model_output = self._main_model(**{k:v for k, v in kw_input.items() if k in keys})
        with torch.no_grad():
            partial_input_dict = {}
            for k, v in kw_input.items():
                if k.startswith("bias_only_") and k[10:] in keys and k[10:] != 'label':
                    partial_input_dict[k[10:]] = v
            if len(partial_input_dict):     # some times we may have no partial input, such as test time
                partial_main_model_output = self._main_model(**partial_input_dict)
            
        if len(partial_input_dict):
            bias_only_model_output = self._bias_only_model(partial_main_model_output['hidden'], kw_input['bias_only_label'])
        else:
            assert not self.training
            bias_only_model_output = {}
        return main_model_output, bias_only_model_output

    @overrides
    def forward(
        self, 
        **kw_input,
    ) -> Dict[str, torch.Tensor]:
        output_dict = super().forward(**kw_input)
        if 'loss' in output_dict and 'bias_only_loss' in output_dict:
            output_dict['loss'] = output_dict['loss'] + self._lambda * output_dict['bias_only_loss']
        return output_dict
