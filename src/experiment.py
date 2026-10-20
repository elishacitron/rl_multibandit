from collections import deque
import pandas as pd
from bandits import Bandit

# TODO: Add rolling average percentage optimal arm
def run_experiment(bandit: Bandit, agent, n_trials: int, window_size: int = 50) -> pd.DataFrame:
    """Returns per-trial results as a DataFrame."""
    recent_rewards = deque(maxlen=window_size)
    optimal_rewards = deque(maxlen=window_size)
    results = []

    for _ in range(n_trials):
        arm = agent.select_action()
        r = bandit.sample(arm)
        agent.update_values(arm, r)

        recent_rewards.append(r)
        rolling_avg_reward = sum(recent_rewards) / len(recent_rewards)

        optimal_arm = bandit.optimal_arm()
        optimal_rewards.append(bandit.r_vals[optimal_arm])
        rolling_avg_optimal_reward = sum(optimal_rewards) / len(optimal_rewards)
        results.append({
            "arm_selected": arm,
            "selected_optimal_arm": arm == optimal_arm,
            "rolling_avg_chosen_rewards": rolling_avg_reward,
            "rolling_avg_optimal_rewards": rolling_avg_optimal_reward
        })

    return pd.DataFrame(results)
