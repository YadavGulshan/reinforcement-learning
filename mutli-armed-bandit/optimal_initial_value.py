import numpy as np
import matplotlib.pyplot as plt


NUM_TRIALS = 1000
EPS = 0
BANDITS_PROB = [0.2, 0.5, 0.75]
np.random.seed(42)

class Bandit:
    def __init__(self, p) -> None:
        self.p = p
        self.p_estimate = 5
        self.N = 1

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = self.p_estimate + (1 / self.N) * (x - self.p_estimate)


def experiment():
    bandits = [Bandit(p) for p in BANDITS_PROB]
    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    for i in range(NUM_TRIALS):
        optimal_j = np.argmax([b.p_estimate for b in bandits])
        num_times_exploited += 1
        j = optimal_j
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)

    for b in bandits:
        print("mean estimate:", b.p_estimate)

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)

    cumulative_reward = np.cumsum(rewards)
    win_rate = cumulative_reward / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rate)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDITS_PROB))
    plt.show()


if __name__ == "__main__":
    experiment()
