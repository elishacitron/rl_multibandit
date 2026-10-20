from agents import EpsilonGreedyAgent
from bandits import StationaryGaussianBandit
from experiment import run_experiment
from plotting import plot_results


def main():
    r_dist = [1.0, 2.0, 3.0, 1.5, 2.5]
    bandit = StationaryGaussianBandit(r_dist, seed=42)
    agent = EpsilonGreedyAgent(n_options=len(r_dist), alpha=0.1, epsilon=0.1, seed=24)

    print("Running trials...")
    rolling_avg_reward = run_experiment(bandit, agent, n_trials=1000)

    print(f"Working reward distribution:\n{agent.action_values}")
    # TODO: the optimal reward is only constant for a stationary bandit
    plot_results(rolling_avg_reward, r_dist[bandit.optimal_arm()])


if __name__ == "__main__":
    main()