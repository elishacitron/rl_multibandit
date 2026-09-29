import random
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

class Bandit:
    def __init__(self,r_dist: list, seed=None):
        self.r_dist = r_dist
        self.n_arms = len(self.r_dist)
        self.rng = random.Random(seed) # self.rng.random() returns a float uniformly in [0.0, 1.0]

    def sample(self, arm: int) -> float:
        p = self.r_dist[arm]
        return 1.0 if self.rng.random() < p else 0.0

    def optimal_arm(self):
        return max(range(self.n_arms), key=lambda a: self.r_dist[a])


class Agent:
    def __init__(self, bandit: Bandit, alpha: float, epsilon: float):
        """Alpha = learning rate; epsilon = random exploration rate"""
        self.bandit = bandit
        self.working_r_dist = [0.0 for _ in range(self.bandit.n_arms)]
        self.alpha = alpha
        self.epsilon = epsilon

    def optimal_arm(self):
        return max(range(self.bandit.n_arms), key=lambda a: self.working_r_dist[a])

    def update_working_r_dist(self, arm: int, r: float):
        self.working_r_dist[arm] += self.alpha * (r - self.working_r_dist[arm])

    def select_action(self):
        if self.bandit.rng.random() < self.epsilon:
            arm = self.bandit.rng.randrange(self.bandit.n_arms)
        else:
            arm = self.optimal_arm()
        r = self.bandit.sample(arm)
        self.update_working_r_dist(arm, r)
        return arm, r

RESULTS_DIR = Path("results")

def main():

    r_dist = [0.1, 0.2, 0.4, 0.3, 0.45]
    bandit = Bandit(r_dist, 42)
    agent = Agent(bandit, alpha=0.1, epsilon=0.1)

    n_trials = 1000
    window_size = 50
    recent_rewards = deque(maxlen=window_size)
    rolling_avg_reward = []

    print("Running trials...")

    for _ in range(n_trials):
        _, r = agent.select_action()
        recent_rewards.append(r)

        rolling_avg_reward.append(sum(recent_rewards) / len(recent_rewards))

    print(f"Working reward distribution:\n{agent.working_r_dist}")

    plt.figure()
    plt.plot(rolling_avg_reward, label="Rolling average reward")
    optimal_reward = bandit.r_dist[bandit.optimal_arm()]
    plt.axhline(optimal_reward, color="red", linestyle="--", label=f"Optimal ({optimal_reward:.2f})")
    plt.xlabel("Trial")
    plt.ylabel("Average Reward")
    plt.title("Trial Results")
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(RESULTS_DIR / f"trial_results_{timestamp}.png")
    plt.show()
    

if __name__ == "__main__":
    main()