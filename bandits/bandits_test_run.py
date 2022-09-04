from bandits.Bandit import Bandit
from bandits.probability_distributions.GaussDistribution import \
    GaussDistribution
from bandits.probability_distributions.BernoulliDistribution import \
    BernoulliDistribution

from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_arms = 10
num_runs = 6000
num_steps = 2000
epsilon = 0.09
gaussian = GaussDistribution(num_arms)
bernoulli = BernoulliDistribution(num_arms)
bandit = Bandit(num_arms, epsilon, bernoulli)

count = 0

print("-----------------------------------")
print("Expected rewards:")
print("-----------------------------------")
for m in bandit.get_mean_values():
    if m == bandit.get_mean_values()[bandit.get_q_star_action()]:
        print("q_star", count, "=", m, " *")
    else:
        print("q_star", count, "=", m)
    count += 1
print("-----------------------------------")

random.seed(4)
all_rewards = []
all_actions = []
best_actions = []

for i in tqdm(range(num_runs)):
    bandit.clear_bandit_state()
    actions, rewards, best_action = bandit.run_experiment(num_steps)
    all_rewards.append(rewards)
    all_actions.append(actions)
    best_actions.append(best_action)

plt.figure(1)
plt.plot(
    [(max(bandit.get_mean_values())) for _ in range(num_steps)],
    linestyle="--"
)
plt.plot(np.mean(all_rewards, axis=0))
plt.yticks([0, .5, 1, 1.5])
plt.title("Average Reward of Epsilon-Greedy Agent")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()

tmp = np.reshape(all_actions, [1, num_runs*num_steps])
series = tmp[0]
plt.figure(2)
pd.Series(series).hist(bins=num_arms)
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.show()

plt.figure(3)
best = np.array(best_actions)
s = np.sum(best, axis=0)/num_runs*100
plt.plot(s)
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xlabel("Steps")
plt.ylabel("Optimal action (%)")
plt.show()
plt.show()
