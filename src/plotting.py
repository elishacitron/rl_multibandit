from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


# TODO: support multiple agents per plot
def plot_results(rolling_avg_reward: list[float], optimal_reward: float) -> None:
    plt.figure()
    plt.plot(rolling_avg_reward, label="Rolling average reward")
    plt.axhline(optimal_reward, color="red", linestyle="--", label=f"Optimal ({optimal_reward:.2f})")
    plt.xlabel("Trial")
    plt.ylabel("Average Reward")
    plt.title("Trial Results")
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(RESULTS_DIR / f"trial_results_{timestamp}.png")
    plt.show()
