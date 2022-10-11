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
from opendebias.models.wrapping_models.base_model import EBDModelBase
from opendebias.modules.losses import EBDLoss
from opendebias.modules.contrastive_loss import ContrastiveLoss
from overrides import overrides
import numpy as np


@Model.register("contrastive_ebd")
class ContrastiveEBDModel(Model):
    def __init__(
        self, 
        vocab: Vocabulary,
        model: Model,
        contrastive_loss: ContrastiveLoss,
        projection_head: FeedForward,
        lambda_contrastive: float,
        lambda_ebd: float,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        model._ret_hidden = True
        self._lambda_contrastive = lambda_contrastive
        self._lambda_ebd = lambda_ebd
        self._model = model
        self._namespace = namespace
        self._label_namespace = label_namespace
        self._projection_head = projection_head
        self._contrastive_loss = contrastive_loss
        self._weights = torch.Tensor(np.load("/data/folder/opendebias/data/bias/mnli-hans/mind-trade-bias/idx2weight.npy"))
        # self._weights = None
        initializer(self)
    
    def _unfold_texts(self, text_dict):
        for key, val in text_dict[self._namespace].items():       # val -> bs x k x seqlen
            text_dict[self._namespace][key] = val.contiguous().view(-1, val.shape[2])
        return text_dict

    def _fold_hidden(self, tensor, bs, k):
        tensor = tensor.contiguous().view(bs, k, tensor.shape[-1])
        return tensor

    @overrides
    def forward(
        self, 
        **kw_input,
    ) -> Dict[str, torch.Tensor]:

        ebd_output = self._model(**kw_input)
        # positive
        # only support tokens as input
        if "positives" not in kw_input:
            return ebd_output
        
        positives, negatives = kw_input["positives"], kw_input["negatives"]
        bs, k_pos, k_neg = positives[self._namespace]['token_ids'].shape[0], positives[self._namespace]['token_ids'].shape[1], negatives[self._namespace]['token_ids'].shape[1]
        # transform shape into bs*k x seqlen
        positives = self._unfold_texts(positives) 
        negatives = self._unfold_texts(negatives)

        # encode instance and get its projection representation
        q_h = ebd_output["hidden"]
        q_z = self._projection_head(q_h)
        with torch.no_grad():
            kp_h = self._model.forward_main_model(tokens=positives)['hidden']
            kn_h = self._model.forward_main_model(tokens=negatives)['hidden']
            kp_z, kn_z = self._projection_head(kp_h), self._projection_head(kn_h)

        # fold, transform h and z to bs x k x z_dim
        kp_h, kn_h = self._fold_hidden(kp_h, bs, k_pos), self._fold_hidden(kn_h, bs, k_neg)
        kp_z, kn_z = self._fold_hidden(kp_z, bs, k_pos), self._fold_hidden(kn_z, bs, k_neg)

        if self._weights is not None:
            # how to get index
            cuda0 = torch.device('cuda')
            index = kw_input['index']
            batch_weights = self._weights[np.array(index.cpu())].to(cuda0)
            contrastive_input_dict = {'q_z': q_z, 'kp_z': kp_z, 'kn_z': kn_z, 'weights':batch_weights}
            ebd_output["loss"] = torch.mean(ebd_output["loss"] / batch_weights)
        else:
            contrastive_input_dict = {'q_z': q_z, 'kp_z': kp_z, 'kn_z': kn_z}
        
        ebd_output["ebd_loss"] = ebd_output["loss"].detach()
        contrastive_loss = self._contrastive_loss(**contrastive_input_dict)
        ebd_output["loss"] = self._lambda_ebd * ebd_output["loss"] + self._lambda_contrastive * contrastive_loss['loss']
        ebd_output.pop("hidden")
        return ebd_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._model.get_metrics(reset=reset)

    def bias_only_model_load_weight(self):
        self._model.bias_only_model_load_weight()