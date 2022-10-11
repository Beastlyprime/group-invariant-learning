import torch

from allennlp.training.learning_rate_schedulers import PolynomialDecay
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
import math


@LearningRateScheduler.register("auto_linear_with_warmup_ratio")
class AutoLinearWithWarmupRatio(PolynomialDecay):
    """
    Implements a learning rate scheduler that increases the learning rate to `lr` during the first
    `warmup_steps` steps, and then decreases it to zero over the rest of the training steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_ratio: float
    ) -> None:
        self.need_reconstruct = True
        self.warmup_ratio = warmup_ratio
        super().__init__(
            optimizer,
            num_epochs=100,
            num_steps_per_epoch=1000,
            power=1.0,
            warmup_steps=1000,
            end_learning_rate=0.0,
            last_epoch=-1,
        )

    
    def construct(self, num_epochs, num_steps_per_epoch):
        warmup_steps = math.ceil((num_epochs*num_steps_per_epoch) * self.warmup_ratio)
        self.warmup_steps = warmup_steps
        self.total_steps = num_epochs * num_steps_per_epoch
        self.steps = 0
        self.step_batch(0)