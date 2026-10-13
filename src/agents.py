import random

# TODO: Make ABC for this module

class Agent:
    def __init__(self, n_arms: int, alpha: float, epsilon: float, seed=None):
        """Alpha = learning rate; epsilon = random exploration rate"""
        self.n_arms = n_arms
        self.action_values = [0.0] * self.n_arms
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def greedy_arm(self):
        return max(range(self.n_arms), key=lambda a: self.action_values[a])

    def update(self, arm: int, r: float):
        self.action_values[arm] += self.alpha * (r - self.action_values[arm])

    def select_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_arms)
        else:
            return self.greedy_arm()