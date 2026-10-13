from collections import deque


# TODO: also record chosen arms, optimal-arm hits and the optimal mean per step
def run_experiment(bandit, agent, n_trials: int, window_size: int = 50) -> list[float]:
    """Returns the rolling average reward per trial."""
    recent_rewards = deque(maxlen=window_size)
    rolling_avg_reward = []

    for _ in range(n_trials):
        arm = agent.select_action()
        r = bandit.sample(arm)
        agent.update(arm, r)
        bandit.step()
        recent_rewards.append(r)
        rolling_avg_reward.append(sum(recent_rewards) / len(recent_rewards))

    return rolling_avg_reward
