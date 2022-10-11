from typing import Dict, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.common import Registrable
from overrides import overrides
from torch.nn import functional as F


class EBDLoss(torch.nn.Module, Registrable):
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@EBDLoss.register("poe")
class ProductOfExperts(EBDLoss):
    @overrides
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if "logits" not in bias_only_model_output:
            return {}
        
        main_logits = main_model_output["logits"]
        bias_logits = bias_only_model_output["logits"]
        ensemble_logits = F.log_softmax(main_logits, 1) + F.log_softmax(bias_logits, 1)  # log(p_i) + log(b_i)
        ensemble_probs = F.softmax(ensemble_logits, 1)  # softmax(log(p_i) + log(b_i))

        output_dict = {
            "probs": ensemble_probs,
            "logits": ensemble_logits
        }
        if labels is not None:
            output_dict['loss'] = F.cross_entropy(ensemble_logits, labels)
        return output_dict

@EBDLoss.register("drift")
class DRiFt(EBDLoss):
    @overrides
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if "logits" not in bias_only_model_output:
            return {}
        
        main_logits = main_model_output["logits"]
        bias_logits = bias_only_model_output["logits"]
        ensemble_logits = main_logits + bias_logits
        ensemble_probs = F.softmax(ensemble_logits, 1) 

        output_dict = {
            "probs": ensemble_probs,
            "logits": ensemble_logits
        }
        if labels is not None:
            output_dict['loss'] = F.cross_entropy(ensemble_logits, labels)
        return output_dict

@EBDLoss.register("learned_mixin")
class LearnedMixin(EBDLoss):
    def __init__(self, input_dim: int, penalty: Optional[float] = None):
        super().__init__()
        self._penalty = penalty
        self._bias_lin = torch.nn.Linear(input_dim, 1)

    @overrides
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if "logits" not in bias_only_model_output:
            return {}
        
        main_logits = main_model_output["logits"]
        bias_logprobs = F.log_softmax(bias_only_model_output["logits"])

        hidden = main_model_output['hidden']
        factor = F.softplus(self._bias_lin.forward(hidden)) # g(x_i)
        bias_logprobs = bias_logprobs * factor  # g(x_i) * log(b_i)

        bias_lp = F.log_softmax(bias_logprobs, 1)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean(0)

        ensemble_logits = F.log_softmax(main_logits, 1) + bias_logprobs  # log(p_i) + g(x_i) * log(b_i)
        ensemble_probs = F.softmax(ensemble_logits, 1)  # softmax(log(p_i) + log(b_i))

        output_dict = {
            "probs": ensemble_probs,
            "logits": ensemble_logits
        }
        if labels is not None:
            loss = F.cross_entropy(ensemble_logits, labels)
            if self._penalty is not None:
                loss += self._penalty * entropy
            output_dict['loss'] = loss
        return output_dict    

@EBDLoss.register("debiased_focal_loss")
class DebiasedFocalLoss(EBDLoss):
    def __init__(self, gamma: float):
        self._gamma = gamma

    @overrides
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        if "logits" not in bias_only_model_output:
            return {}
        
        main_logits = main_model_output["logits"]
        output_dict = {
            "probs": F.softmax(main_logits),
            "logits": main_logits
        }
        if labels is not None:
            loss = F.cross_entropy(main_logits, labels, reduction='none')
            bias_probs = F.softmax(bias_only_model_output["logits"])
            weights = torch.gather(bias_probs, 1, labels.unsqueeze(-1)).squeeze(-1)
            weights = (1-weights) ** self._gamma
            weighted_loss = (weights * loss).sum() / loss.shape[0]
            output_dict['loss'] = weighted_loss
        return output_dict


@EBDLoss.register("cross_entropy")
class PlainCrossEntropy(EBDLoss):
    @overrides
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None,
                is_vector=False) -> Dict[str, torch.Tensor]:
        if labels is None:
            return {}
        main_logits = main_model_output["logits"]
        if is_vector:
            loss = F.cross_entropy(main_logits, labels, reduction='none')
        else:
            loss = F.cross_entropy(main_logits, labels)
        output_dict = {
            "loss": loss,
            "logits": main_logits,
            "probs": F.softmax(main_logits)
        }
        return output_dict

@EBDLoss.register("weighted_cross_entropy")
class PlainCrossEntropy(EBDLoss):
    @overrides
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if labels is None:
            return {}
        main_logits = main_model_output["logits"]
        if self.training:
            loss = F.cross_entropy(main_logits, labels, reduction='none')
        else:
            loss = F.cross_entropy(main_logits, labels)
        output_dict = {
            "loss": loss,
            "logits": main_logits,
            "probs": F.softmax(main_logits)
        }
        return output_dict

@EBDLoss.register("conf_reg")
class ConfidenceRegularization(EBDLoss):
    @overrides
    def forward(self, 
                main_model_output: Dict[str, object], 
                bias_only_model_output: Dict[str, object], 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if "logits" not in bias_only_model_output:
            return {}
        
        main_logits = main_model_output["logits"]
        output_dict = {
            "probs": F.softmax(main_logits),
            "logits": main_logits
        }

        if labels is not None:
            log_probs = F.log_softmax(main_logits, dim=-1)
            teacher_probs = F.softmax(bias_only_model_output["logits"])
            loss = -torch.sum(teacher_probs * log_probs, dim=-1)
            loss = loss.mean()
            output_dict['loss'] = loss
        return output_dict
