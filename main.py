import random

class Bandit:
    def __init__(self, r_dist, seed=None):
        self.r_dist = list(r_dist)
        self.n_arms = len(self.r_dist)
        self.rng = random.Random(seed)

    def sample(self, arm) -> float:
        p = self.r_dist[arm]
        return 1.0 if self.rng.random() < p else 0.0

    def optimal_arm(self) -> int:
        return max(range(self.n_arms), key=lambda a: self.r_dist[a])

class Agent:
    pass

