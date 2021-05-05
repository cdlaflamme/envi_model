#scheduler.py
#neural network learning rate scheduler. contains an optimizer; changes its base learning rate according to some scheme.

from enum import Enum

class Schedule(Enum):
    STATIC = 0
    EXP = 1
    STEP = 2

class Scheduler:
    def __init__(self, model_size, factor, warmup, optimizer, mode=Schedule.EXP, exp_decay=0.5, step_count=560, static_lr = 1e-7):
        self.optimizer = optimizer
        self.mode = mode
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.exp_decay = abs(exp_decay)
        self.step_count = step_count
        self.base_rate = (self.model_size ** (-0.5)) * self.factor
        self.static_lr = static_lr
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `rate` above"
        if step is None:
            step = self._step
        
        warmup_factor = step * self.warmup ** (-1.5)
        #Exponential decay
        if self.mode==Schedule.EXP:
            mode_factor = step ** (-self.exp_decay)
        #Step decay
        elif self.mode==Schedule.STEP:
            mode_factor = ((self.exp_decay) ** math.floor(step / self.step_count))
        #No decay (static learning rate)
        elif self.mode==Schedule.STATIC:
            return self.static_lr
        #return calculated rate if static not used
        return self.base_rate * min(warmup_factor, mode_factor)
        