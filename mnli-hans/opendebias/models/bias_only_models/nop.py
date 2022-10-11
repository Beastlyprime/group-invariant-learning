import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from overrides import overrides
import torch.nn.functional as F

from allennlp.models import Model

logger = logging.getLogger(__name__)


@Model.register("no-op")
class NonOpBiasOnly(Model):
    def forward(self):
        return {}