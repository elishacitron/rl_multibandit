import random
from abc import ABC, abstractmethod

DRIFT_AMPLITUDE_MEAN = 0
DRIFT_AMPLITUDE_SD = 5
DRIFT_FREQ_MEAN = 0
DRIFT_FREQ_SD = 5

class Bandit(ABC):
    def __init__(self, r_vals: list, seed=None) -> None:
        self.r_vals = r_vals
        self.n_arms = len(self.r_vals)
        self.rng = random.Random(seed)

    def optimal_arm(self) -> int:
        return max(range(self.n_arms), key=lambda a: self.r_vals[a])

    @abstractmethod
    def sample(self, arm:int) -> float:
        pass


class StationaryGaussianBandit(Bandit):
    def sample(self, arm: int) -> float:
        return self.rng.gauss(self.r_vals[arm], 1.0)


class StationaryBinaryBandit(Bandit):
    def sample(self, arm: int) -> float:
        p = self.r_vals[arm]
        return 1.0 if self.rng.random() < p else 0.0


class OscillatingBandit(Bandit):
    # TODO: finish drift(), call it from step()
    def drift(self):
        for arm_idx in range(self.n_arms):
            a = abs(random.gauss(DRIFT_AMPLITUDE_MEAN, DRIFT_AMPLITUDE_SD)) # Drift amplitude
            b = abs(random.gauss(DRIFT_FREQ_MEAN, DRIFT_FREQ_SD)) # Drift frequency
            # What if I get 0 for either of the above?

        # Need a function for each arm
        # Function should be made of a superposition of sin waves, parameterised randomly from some distribution
            # What's the best way of storing this function?
        # Then, at each step (trial), update each arm's r by taking some step 'along' the function

    def step(self) -> None:
        # TODO: advance time and update self.r_vals via drift()
        pass

    def sample(self, arm: int) -> None:
        pass