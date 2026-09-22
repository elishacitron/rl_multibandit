import random

class Bandit:
    def __init__(self,r_dist: list, seed=None):
        self.r_dist = r_dist
        self.n_arms = len(self.r_dist)
        self.rng = random.Random(seed)
        # self.rng.random() returns a float uniformly in [0.0, 1.0]

    def sample(self, arm: int) -> float:
        p = self.r_dist[arm]
        return 1.0 if self.rng.random() < p else 0.0


class Agent:
    def __init__(self, bandit: Bandit, alpha: float, epsilon: float, seed=None):
        self.bandit = bandit
        self.n_arms = bandit.n_arms
        self.working_r_dist = [0.0 for _ in range(self.n_arms)]
        self.alpha = alpha
        self.epsilon = epsilon

    def optimal_arm(self):
        return max(range(self.n_arms), key=lambda a: self.working_r_dist[a])

    def update_working_r_dist(self, arm: int, r: float):
        self.working_r_dist[arm] += self.alpha * (r - self.working_r_dist[arm])

    def select_action(self):
        if random.random() < self.epsilon:
            arm = random.randrange(self.n_arms)
        else:
            arm = self.optimal_arm()
        r = self.bandit.sample(arm)
        self.update_working_r_dist(arm, r)
        return arm, r

def main():

    r_dist = [0.1, 0.4, 0.8]
    bandit = Bandit(r_dist, 42)
    agent = Agent(bandit, alpha=0.1, epsilon=0.1, seed=42)

if __name__ == "__main__":
    main()