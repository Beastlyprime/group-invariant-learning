from typing import Dict, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.common import Registrable
from overrides import overrides
from torch.nn import functional as F


class ContrastiveLoss(torch.nn.Module, Registrable):
    def forward(self, q_z: torch.Tensor, kp_z: torch.Tensor, kn_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@ContrastiveLoss.register("info-nce")
class InfoNCE(ContrastiveLoss):
    def __init__(self, tau, full_pos_denominator=False):
        super().__init__()
        self._tau = tau
        self._full_pos_denominator = full_pos_denominator
    
    @overrides
    def forward(self, q_z: torch.Tensor, kp_z: torch.Tensor, kn_z: torch.Tensor, weights: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # normalize z vector
        q_z = q_z / torch.norm(q_z, p=2, dim=-1).unsqueeze(-1)  # b x z
        kp_z = kp_z / torch.norm(kp_z, p=2, dim=-1).unsqueeze(-1)  # b x k x z
        kn_z = kn_z / torch.norm(kn_z, p=2, dim=-1).unsqueeze(-1)  # b x k x z

        q_z = q_z.unsqueeze(1)  # b x 1 x z

        q_kp_dot_exp = torch.exp(torch.matmul(q_z, kp_z.transpose(1,2)).squeeze(1) / self._tau)    # b x kp
        q_kn_dot_exp = torch.exp(torch.matmul(q_z, kn_z.transpose(1,2)).squeeze(1) / self._tau)   # b x kn
        q_kn_dot_exp_sum = torch.sum(q_kn_dot_exp, dim=-1).unsqueeze(-1) # b x 1

        if not self._full_pos_denominator:
            denominator = q_kn_dot_exp_sum + q_kp_dot_exp # b x kp
        else:
            denominator = q_kn_dot_exp_sum + torch.sum(q_kp_dot_exp, dim=-1).unsqueeze(-1)  # b x 1    use all positive pair as denominator
        loss_per_instance = -torch.mean(torch.log(q_kp_dot_exp / denominator), dim=-1)    # b
        if weights is None:
            loss = torch.mean(loss_per_instance)
        else:
            loss = torch.mean(loss_per_instance / weights)

        output_dict = {
            'loss': loss
        }
        return output_dict
