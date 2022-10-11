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


@Model.register("two_stage")
class TwoStageEBDModel(EBDModelBase):

    @overrides
    def interaction(self,
                    **kw_input) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        main_model_output = self._main_model(**{k:v for k, v in kw_input.items() if k in inspect.getfullargspec(self._main_model.forward).args and k != "label"})
        bias_only_input_dict = {}
        bias_only_args = inspect.getfullargspec(self._bias_only_model.forward).args
        # check whether is combined dataset, and pack the bias-only model input
        combined_dataset = False
        for k in kw_input.keys():
            if k.startswith('bias_only_'):
                combined_dataset = True
                break 
        for k, v in kw_input.items():
            if combined_dataset:
                if k.startswith('bias_only_') and k[10:] in bias_only_args:
                    bias_only_input_dict[k[10:]] = v 
            else:
                if k in bias_only_args and k != "label":
                    bias_only_input_dict[k] = v
        with torch.no_grad():
            bias_only_model_output = self._bias_only_model(**bias_only_input_dict)
        return main_model_output, bias_only_model_output
