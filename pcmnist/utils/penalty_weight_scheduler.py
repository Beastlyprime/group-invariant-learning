class PenaltyWeightScheduler:
    def __init__(self, iter_to_max, init_val, max_val):
        assert iter_to_max >= 0
        self.iter_to_max = iter_to_max
        self.init_val = init_val
        self.max_val = max_val
        self.step_val = (self.max_val - self.init_val) / self.iter_to_max if self.iter_to_max > 0 else 0
        # self.jump = jump


    def step(self, iter_n):
        if iter_n < 0: 
            return self.init_val
        elif iter_n >= self.iter_to_max:
            return self.max_val
        elif self.max_val > 1.0:
            return 1.0
        else:
            return self.init_val + self.step_val * iter_n