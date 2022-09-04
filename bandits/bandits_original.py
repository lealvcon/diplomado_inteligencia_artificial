# Diplomado de Inteligencia Artificial Aplicada
# GCIC-IPICYT
# Salvador Ruiz Correa (You-i Lab)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

class bandits():
    def __init__(self, n_arms):
        """
        Arguments:
        """
        self.n_arms = n_arms
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
        self.epsilon = .1

    def set_mean_values(self):
        self.mean_values = [random.gauss(0,1) for i in range(self.n_arms)]
        self.q_star_action = np.argmax(self.mean_values)


    def clear_agent(self):
        self.Q[:] = 0
        self.N[:] = 0

    def bandit(self,A):
        return random.gauss(self.mean_values[A],1)

    def argmax(self):
        """
        Takes in a list of Q values and returns the index
        of the item with the highest value. Breaks ties randomly.
        returns: int - the index of the highest value in Q
        """
        top = float("-inf")
        ties = []

        for i in range(self.n_arms):
            if self.Q[i] > top:
                top, ties = self.Q[i], [i]
            elif self.Q[i] == top:
                ties.append(i)
        index = np.random.choice(ties)
        return index

    #    def arg_max(Q, N, t):
    #       return np.random.choice(np.flatnonzero(Q == Q.max())) # breaking ties randomly


    def epsilon_greedy(self, epsilon):
        self.epsilon = epsilon
        if np.random.random() < self.epsilon:
            current_action = np.random.randint(self.n_arms)
        else:
            current_action = self.argmax()
        return current_action

    def update(self, A, R):
        self.N[A] = self.N[A] + 1;
        self.Q[A] = self.Q[A] + 1/self.N[A]*(R-self.Q[A])

    def agent(self, num_iter, epsilon):
        actions =[]
        rewards = []
        best_action = []
        for iter in range(0,num_iter):
            A = self.epsilon_greedy(epsilon)
            if A == self.q_star_action:
                best_action.append(1)
            else:
                best_action.append(0)
            R = self.bandit(A)
            actions.append(A);
            rewards.append(R)
            self.update(A,R)
        return actions, rewards, best_action


# k-bandit environment
num_arms = 10
bd = bandits(num_arms);
bd.set_mean_values()
count = 0

print("-----------------------------------")
print("Expected rewards:")
print("-----------------------------------")
for m in bd.mean_values:
    if m == bd.mean_values[bd.q_star_action]:
        print("q_star", count, "=", m, " *")
    else:
        print("q_star", count, "=", m)
    count += 1
print("-----------------------------------")


# k-bandit environment
random.seed(4)
all_rewards = []
all_actions = []
best_actions = []
num_runs = 6000
num_steps = 2000
epsilon = 0.09
for i in tqdm(range(num_runs)):
    bd.clear_agent()
    actions, rewards,best_action = bd.agent(num_steps, epsilon)
    all_rewards.append(rewards)
    all_actions.append(actions)
    best_actions.append(best_action)

plt.plot([(max(bd.mean_values)) for _ in range(num_steps)], linestyle="--")
plt.plot(np.mean(all_rewards, axis=0))
plt.yticks([0, .5, 1, 1.5])
plt.title("Average Reward of Epsilon-Greedy Agent")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()

tmp = np.reshape(all_actions,[1,num_runs*num_steps])
series = tmp[0]
pd.Series(series).hist(bins=num_arms);
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.show()


best = np.array(best_actions)
s=np.sum(best,axis=0)/num_runs*100
plt.plot(s)
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xlabel("Steps")
plt.ylabel("Optimal action (\%)")
plt.show()
plt.show()
