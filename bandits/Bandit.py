import numpy as np


class Bandit:
    """
    Simulates a k-armed bandit experiment

    Params
    -----------
    number_of_arms : int
        The `k` in 'k-armed bandit'.
    epsilon : float
        The probability to explore actions instead of taking a
        greedy action.
    probability_dist : ProbabilityDistribution
        A class that implements the probability distribution used by the
        bandits.

    Notes
    -----

    After each choice among the k options you expect a mean reward `R` given
    that that action was selected (value of that action). The action selected
    at a time `t` is named `At` with its corresponding reward `Rt`.

    The value of an Action is denoted as `q*(a)` or (q star of `a`), and we
    denote the estimated value of an action `a` at a time step `t` as `Qt(a)`.
    We would like `Qt(a)` to be close to `q*(a)` since we do not know the
    actions values with certainty.
    """

    def __init__(self, number_of_arms, epsilon, probability_dist,
                 optimistic_initial_value=0):
        self.__number_of_arms = number_of_arms
        self.__epsilon = epsilon
        self.__probability_dist = probability_dist
        self.__optimistic_initial_value = optimistic_initial_value
        # Estimated value `Qt(a)`
        self.Q = np.zeros(self.__number_of_arms)
        # Number of times each arm is selected
        self.N = np.zeros(self.__number_of_arms)
        self.__mean_values = None
        self.__q_star_action = None
        self.__set_mean_values()

    def get_number_of_arms(self):
        return self.__number_of_arms

    def get_epsilon(self):
        return self.__epsilon

    def get_mean_values(self):
        return self.__mean_values

    def get_q_star_action(self):
        return self.__q_star_action

    def probability_distribution(self):
        return self.__probability_dist

    def __set_mean_values(self):
        self.__mean_values = \
            self.probability_distribution().set_mean_values()
        self.__q_star_action = np.argmax(self.__mean_values)

    def clear_bandit_state(self):
        self.Q[:] = 0 + self.__optimistic_initial_value
        self.N[:] = 0

    def _pull_lever(self, action):
        mean_reward = self.get_mean_values()[action]
        return self.probability_distribution().get_value(mean_reward)

    def __argmax(self):
        """
        Takes in a list Q of k values and returns the index
        of the item with the highest value. Breaks ties randomly.
        returns: int - the index of the highest value in Q
        """
        top = float("-inf")
        ties = []

        for i in range(self.get_number_of_arms()):
            if self.Q[i] > top:
                top, ties = self.Q[i], [i]
            elif self.Q[i] == top:
                ties.append(i)
        index = np.random.choice(ties)
        return index

    # def arg_max(Q, N, t):
    # # breaking ties randomly
    #   return np.random.choice(np.flatnonzero(Q == Q.max()))

    def __generate_action(self):
        """
        Determines if the algorithm should take a greedy action (take the
        highest estimated value) or explore the possible actions available.

        Notes
        -----

        A simple alternative is to behave greedily most of the time,
        but sometimes, with small probability `epsilon` we select randomly
        from among all the actions with equal probability (explore)

        """
        return self.__argmax() if self._is_greedy() else \
            self._explore_actions()

    def _is_greedy(self):
        """
        Calculates the probability of taking a greedy action, given that the
        probability of exploring is `epsilon` (random < epsilon)
        """
        return np.random.random() >= self.get_epsilon()

    def _explore_actions(self):
        return np.random.randint(self.get_number_of_arms())

    def _update_experiment(self, action, reward):
        self.N[action] = self.N[action] + 1
        self.Q[action] = \
            self.Q[action] + 1/self.N[action] * (reward - self.Q[action])

    def run_experiment(self, num_choices):
        actions = []
        rewards = []
        best_action = []
        for choice in range(0, num_choices):
            action = self.__generate_action()
            if action == self.get_q_star_action():
                best_action.append(1)
            else:
                best_action.append(0)
            reward = self._pull_lever(action)
            actions.append(action)
            rewards.append(reward)
            self._update_experiment(action, reward)
        return actions, rewards, best_action
