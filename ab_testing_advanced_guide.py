import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

class ABTesting:
    def __init__(self, control_data, treatment_data):
        self.control_data = control_data
        self.treatment_data = treatment_data

    def calculate_effect_size(self):
        control_mean = np.mean(self.control_data)
        treatment_mean = np.mean(self.treatment_data)
        return treatment_mean - control_mean

    def power_analysis(self, alpha=0.05, power=0.8, delta=None, sigma=None):
        if delta is None:
            delta = self.calculate_effect_size()
        if sigma is None:
            sigma = np.std(np.concatenate([self.control_data, self.treatment_data]))
        # Calculating sample size required for given power
        n = int((stats.norm.ppf(1-alpha/2) + stats.norm.ppf(power)) ** 2 * (2 * sigma ** 2) / delta ** 2)
        return n

    def randomization(self):
        np.random.shuffle(self.control_data)
        np.random.shuffle(self.treatment_data)

    def sanity_checks(self):
        assert len(self.control_data) > 0, 'Control group is empty'
        assert len(self.treatment_data) > 0, 'Treatment group is empty'

    def hypothesis_testing(self, alpha=0.05):
        t_stat, p_value = stats.ttest_ind(self.control_data, self.treatment_data)
        return p_value < alpha

    def business_impact_analysis(self):
        effect_size = self.calculate_effect_size()
        # Assuming we have a baseline of some business metric (e.g., revenue)
        baseline_revenue = 100000  # Example baseline revenue
        projected_revenue = baseline_revenue + (effect_size * baseline_revenue)
        return projected_revenue

# Example usage:
# control = [120, 130, 115, 140, 150]
# treatment = [135, 145, 140, 155, 150]
# ab_test = ABTesting(control, treatment)
# ab_test.sanity_checks()
# print('Effect Size:', ab_test.calculate_effect_size())
# print('Power Analysis Sample Size:', ab_test.power_analysis())
# print('Hypothesis Testing:', ab_test.hypothesis_testing())
# print('Projected Revenue Impact:', ab_test.business_impact_analysis())
