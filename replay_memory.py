from experience import Experience
import random
import numpy as np

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
    
    def get_sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    @staticmethod
    def unzip_sample(sample):
        states = np.array([s.state for s in sample])
        actions = np.array([s.action for s in sample])
        new_states = np.array([s.new_state for s in sample])
        rewards = np.array([s.reward for s in sample])
        dones = np.array([s.done for s in sample])
        return states, actions, new_states, rewards, dones


if __name__ == '__main__':
    memory = ReplayMemory(10)
    memory.push(Experience(1, 2, 3, 4, 5))
    memory.push(Experience(1, 2, 3, 4, 5))
    memory.push(Experience(1, 2, 3, 4, 5))
    memory.push(Experience(1, 2, 3, 4, 5))
    sample = memory.get_sample(3)
    print(ReplayMemory.unzip_sample(sample))