import random
from abc import ABC, abstractmethod

DRIFT_AMPLITUDE_MEAN = 0
DRIFT_AMPLITUDE_SD = 5
DRIFT_FREQ_MEAN = 0
DRIFT_FREQ_SD = 5

class Bandit(ABC):
    def __init__(self, initial_r_vals: list, seed=None) -> None:
        self.initial_r_vals = initial_r_vals
        self.n_arms = len(self.initial_r_vals)
        self.rng = random.Random(seed)

    @abstractmethod
    def sample(self, arm:int) -> float:
        pass

    @abstractmethod
    def optimal_arm(self) -> int:
        pass

    @abstractmethod
    def step(self) -> None:
        pass


class StationaryBandit(Bandit):
    def sample(self, arm: int) -> float:
        return self.rng.gauss(self.initial_r_vals[arm], 1.0)

    def optimal_arm(self) -> int:
        return max(range(self.n_arms), key=lambda a: self.initial_r_vals[a])

    def step(self) -> None:
        pass


class OscillatingBandit(Bandit):
    def __init__(self, initial_r_vals: list, seed=None):
        super().__init__(initial_r_vals, seed)
        self.r_dist = list(initial_r_vals)  # current means; drift should update these

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
        # TODO: advance time and update self.r_dist via drift()
        pass

    def sample(self, arm: int) -> float:
        return self.rng.gauss(self.r_dist[arm], 1.0)

    def optimal_arm(self) -> int:
        return max(range(self.n_arms), key=lambda a: self.r_dist[a])