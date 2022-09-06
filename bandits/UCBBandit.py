from bandits.Bandit import Bandit
import numpy as np
import math


class UCBBandit(Bandit):

    def __init__(self, number_of_arms, epsilon, probability_dist,
                 confidence_level,
                 optimistic_initial_value=0):
        self.__confidence_level = confidence_level
        super().__init__(
            number_of_arms,
            epsilon,
            probability_dist,
            optimistic_initial_value
        )

    def __argmax(self, num_choice):
        top = float("-inf")
        ties = []

        for i in range(self.get_number_of_arms()):
            argument = self.__calculate_ucb_argument(i, num_choice)
            if argument > top:
                top, ties = argument, [i]
            elif argument == top:
                ties.append(i)
        index = np.random.choice(ties)
        return index

    def __calculate_ucb_argument(self, action, num_choice):
        value = 0
        if self.N[action] == 0:
            value = self.Q[action]
        else:
            value = self.Q[action] + \
                self.__confidence_level * math.sqrt(
                    math.log(num_choice) / self.N[action]
                )

        return value

    def __generate_action(self, num_choice):
        """
        Determines if the algorithm should take a greedy action (take the
        highest estimated value) or explore the possible actions available.

        Notes
        -----

        A simple alternative is to behave greedily most of the time,
        but sometimes, with small probability `epsilon` we select randomly
        from among all the actions with equal probability (explore)

        """
        return self.__argmax(num_choice) if self._is_greedy() else \
            self._explore_actions()

    def run_experiment(self, num_choices):
        actions = []
        rewards = []
        best_action = []
        for choice in range(0, num_choices):
            action = self.__generate_action(num_choices)
            if action == self.get_q_star_action():
                best_action.append(1)
            else:
                best_action.append(0)
            reward = self._pull_lever(action)
            actions.append(action)
            rewards.append(reward)
            self._update_experiment(action, reward)
        return actions, rewards, best_action
