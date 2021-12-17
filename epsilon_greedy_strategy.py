import math

class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay) -> None:
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, step):
        return self.end + (self.start - self.end) * math.exp(-1. * step * self.decay)