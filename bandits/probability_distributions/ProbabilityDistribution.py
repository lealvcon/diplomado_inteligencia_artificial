
class ProbabilityDistribution:

    def __init__(self, number_of_arms):
        self.number_of_arms = number_of_arms

    def get_number_of_arms(self):
        return self.number_of_arms

    def get_value(self, mean_reward):
        raise NotImplementedError

    def set_mean_values(self, number_of_arms):
        raise NotImplementedError
