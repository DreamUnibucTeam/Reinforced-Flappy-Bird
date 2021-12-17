import random
import numpy as np

class Agent:
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
    
    def _get_action_from_dqn(self, dqn, diff):
        state_for_prediction = np.expand_dims(diff, axis=0)
        predictions = dqn(state_for_prediction, training=False)
        return np.argmax(predictions)

    def select_action(self, diff, dqn):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions) # explore
        else:
            return self._get_action_from_dqn(dqn, diff)