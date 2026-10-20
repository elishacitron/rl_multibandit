import random
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, n_options: int, seed=None):
        self.n_options = n_options
        self.action_values = [0.0] * self.n_options
        self.rng = random.Random(seed)

    @abstractmethod
    def update_values(self, arm: int, r: float):
        pass

    @abstractmethod
    def select_action(self):
        pass

class EpsilonGreedyAgent(Agent):
    def __init__(self, n_options: int, alpha: float, epsilon: float, seed = None):
        super().__init__(n_options, seed)
        self.alpha = alpha
        self.epsilon = epsilon

    def update_values(self, arm: int, r: float): # TODO: Separate value updating from policy - look at CoPilot chat from 20/10
        self.action_values[arm] += self.alpha * (r - self.action_values[arm])

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_options)
        else:
            return self.greedy_arm()

    def greedy_arm(self) -> int:
        max_r = max(self.action_values)
        greedy_options = [o for o in range(self.n_options) if self.action_values[o] == max_r]
        if len(greedy_options) == 1:
            return greedy_options[0]
        else:
            return greedy_options[self.rng.randrange(len(greedy_options))]