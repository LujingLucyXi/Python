import numpy as np
import pandas as pd
import scipy.stats as stats
import math

class ABTestingFramework:
    def __init__(self, conversion_rate_A, conversion_rate_B, sample_size, alpha=0.05, beta=0.2):
        self.conversion_rate_A = conversion_rate_A
        self.conversion_rate_B = conversion_rate_B
        self.sample_size = sample_size
        self.alpha = alpha
        self.beta = beta
        self.sample_A = []
        self.sample_B = []

    def randomization(self):
        self.sample_A = np.random.binomial(1, self.conversion_rate_A, self.sample_size)
        self.sample_B = np.random.binomial(1, self.conversion_rate_B, self.sample_size)

    def sanity_checks(self):
        if not (0 <= self.conversion_rate_A <= 1 and 0 <= self.conversion_rate_B <= 1):
            raise ValueError("Conversion rates must be between 0 and 1")

    def hypothesis_testing(self):
        obs_A = np.mean(self.sample_A)
        obs_B = np.mean(self.sample_B)
        std_A = np.std(self.sample_A, ddof=1)
        std_B = np.std(self.sample_B, ddof=1)
        z_score = (obs_A - obs_B) / np.sqrt((std_A**2 / self.sample_size) + (std_B**2 / self.sample_size))
        p_value = stats.norm.sf(abs(z_score)) * 2
        return z_score, p_value

    def calculate_power(self):
        p1 = self.conversion_rate_A
        p2 = self.conversion_rate_B
        effect_size = p2 - p1
        delta = effect_size * (self.sample_size ** 0.5) / np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        power = stats.norm.cdf(delta - stats.norm.ppf(1 - self.alpha))
        return power

    def business_impact_analysis(self):
        return (self.conversion_rate_B - self.conversion_rate_A) * self.sample_size

if __name__ == '__main__':
    a_b_test = ABTestingFramework(conversion_rate_A=0.10, conversion_rate_B=0.15, sample_size=1000)
    a_b_test.sanity_checks()
    a_b_test.randomization()
    z_score, p_value = a_b_test.hypothesis_testing()
    power = a_b_test.calculate_power()
    business_impact = a_b_test.business_impact_analysis()

    print(f"Z-Score: {z_score}, P-Value: {p_value}, Power: {power}, Business Impact: {business_impact}")