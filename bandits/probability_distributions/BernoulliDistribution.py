from bandits.probability_distributions.ProbabilityDistribution import \
    ProbabilityDistribution
import random
import numpy


class BernoulliDistribution(ProbabilityDistribution):

    def __init__(self, number_of_arms):
        super().__init__(number_of_arms)

    def get_value(self, mean_reward):
        return numpy.random.binomial(1, mean_reward) * mean_reward

    def set_mean_values(self):
        return [random.random() for _ in range(self.get_number_of_arms())]
