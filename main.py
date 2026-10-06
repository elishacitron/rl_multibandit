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

RESULTS_DIR = Path("results")

def main():

    r_dist = [0.1, 0.2, 0.4, 0.3, 0.6]
    bandit = Bandit(r_dist, seed=42)
    agent = Agent(n_arms=len(r_dist), alpha=0.1, epsilon=0.1, seed=24)

    n_trials = 1000
    window_size = 50
    recent_rewards = deque(maxlen=window_size)
    rolling_avg_reward = []

    print("Running trials...")

    for _ in range(n_trials):
        arm = agent.select_action()
        r = bandit.sample(arm)
        agent.update(arm, r)
        recent_rewards.append(r)

        rolling_avg_reward.append(sum(recent_rewards) / len(recent_rewards))

    print(f"Working reward distribution:\n{agent.action_values}")

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