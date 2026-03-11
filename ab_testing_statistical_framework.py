import numpy as np
import scipy.stats as stats
import pandas as pd

# Constants
ALPHA = 0.05
POWER = 0.80
MVP_PERCENT = 0.05

# Function for calculating sample size for two-proportion z-test

def calculate_sample_size(p1, p2, alpha, power):
    # Calculate the effect size 
    effect_size = abs(p1 - p2)
    # Calculate sample size
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2) * ((p1 * (1 - p1) + p2 * (1 - p2)) / (effect_size ** 2))
    return int(np.ceil(n))

# Function for randomization

def randomize_groups(data):
    # Shuffle data and create two groups
    np.random.shuffle(data)
    return data[:len(data)//2], data[len(data)//2:]

# Sanity checks for input data

def sanity_checks(control_group, treatment_group):
    # Check if mean values are plausible
    assert control_group.mean() >= 0 and treatment_group.mean() >= 0,
    "Control and treatment means must be non-negative."
    # Further checks can be added here

# Hypothesis testing using two-proportion z-test

def hypothesis_testing(control_events, control_n, treatment_events, treatment_n, alpha):
    # Calculate proportions
    p_control = control_events / control_n
    p_treatment = treatment_events / treatment_n
    # Conduct z-test
    z_stat = (p_treatment - p_control) / np.sqrt((p_control * (1 - p_control) / control_n) + (p_treatment * (1 - p_treatment) / treatment_n))
    p_value = 1 - stats.norm.cdf(z_stat)
    return p_value

# Business impact analysis

def business_impact(control_n, treatment_n, revenue_per_user):
    control_revenue = control_n * revenue_per_user
    treatment_revenue = treatment_n * revenue_per_user
    impact = treatment_revenue - control_revenue
    return impact


# Example analysis

if __name__ == '__main__':
    # Simulated data for control and treatment
    control_data = np.random.binomial(1, 0.5, size=1000)
    treatment_data = np.random.binomial(1, 0.55, size=1000)

    # Calculate sample size
    sample_size = calculate_sample_size(0.5, 0.55, ALPHA, POWER)
    print(f'Required sample size for each group: {sample_size}')
    
    # Randomize groups
    control_group, treatment_group = randomize_groups(control_data)
    
    # Sanity checks
    sanity_checks(control_group, treatment_group)
    
    # Hypothesis testing
    control_events = np.sum(control_group)
    treatment_events = np.sum(treatment_group)
    p_value = hypothesis_testing(control_events, len(control_group), treatment_events, len(treatment_group), ALPHA)
    print(f'P-value: {p_value}')

    # Business impact analysis
    revenue_per_user = 100  # Assuming each user generates $100
    impact = business_impact(len(control_group), len(treatment_group), revenue_per_user)
    print(f'Business Impact: ${impact}')