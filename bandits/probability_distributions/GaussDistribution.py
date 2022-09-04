from bandits.probability_distributions.ProbabilityDistribution import \
    ProbabilityDistribution
import random


class GaussDistribution(ProbabilityDistribution):

    def __init__(self, number_of_arms):
        super().__init__(number_of_arms)

    def get_value(self, mean_reward):
        return random.gauss(mean_reward, 1)

    def set_mean_values(self):
        return [
            random.gauss(0, 1) for _ in range(self.get_number_of_arms())
        ]
