import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Configuration
significance_level = 0.05  # 5% significance level
power = 0.8  # 80% power
minimum_effect_size = 0.1  # Example minimum effect size

# Sample Size Calculation using Power Analysis
# Assuming proportions for conversion rates in the control and treatment groups
p1 = 0.1  # Conversion rate in control group
p2 = p1 + minimum_effect_size  # Expected conversion rate in treatment group

# Calculate sample size needed for each group
required_sample_size = int(np.ceil(stats.proportion.bisect(0.9,
    [p1, p2], alpha=significance_level, power=power)))

# Random Stratified Assignment to Control/Treatment Groups
np.random.seed(42)  # For reproducibility
sample = pd.DataFrame({'id': range(1, required_sample_size + 1)})
# Shuffle and assign to groups
sample['group'] = np.where(np.random.rand(required_sample_size) < 0.5, 'control', 'treatment')

# Sanity Checks: Sample Ratio Mismatch (SRM) Test
control_count = (sample['group'] == 'control').sum()
treatment_count = (sample['group'] == 'treatment').sum()

# Chi-square test for Sample Ratio Mismatch
observed_counts = [control_count, treatment_count]
expected_counts = [required_sample_size / 2] * 2
chi2_stat = stats.chisquare(observed_counts, f_exp=expected_counts)[0]

# Data Validation
assert (control_count + treatment_count) == required_sample_size, "Total sample size mismatch"

# Group Balance Verification
print("Control Count:", control_count)
print("Treatment Count:", treatment_count)

# Metric Collection Simulation for 5% MVP Rollout
mvp_sample_size = int(required_sample_size * 0.05)
mvp_sample = sample.sample(n=mvp_sample_size)
mvp_sample['conversion'] = np.random.binomial(1, p1, size=mvp_sample_size)  # Simulated conversion rates

# Statistical Tests
# Chi-square for conversion rates
contingency_table = pd.crosstab(mvp_sample['group'], mvp_sample['conversion'])
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

# Independent t-test for continuous metrics (simulated)
metric_control = np.random.normal(loc=10, scale=2, size=control_count)
metric_treatment = np.random.normal(loc=10 + minimum_effect_size, scale=2, size=treatment_count)
independent_t_test = stats.ttest_ind(metric_control, metric_treatment)

# Effect Size Calculation (Cohen's h for proportions)
cohen_h = (p2 - p1) / np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

# Confidence Interval Computation
confidence_interval = stats.norm.interval(0.95, loc=np.mean(mvp_sample['conversion']),
    scale=stats.sem(mvp_sample['conversion']))

# Hypothesis Testing Decision Logic
if p_value < significance_level:
    decision = "Reject Null Hypothesis"
else:
    decision = "Fail to Reject Null Hypothesis"

# Reporting
print(f"Chi-squared: {chi2}, p-value: {p_value}")
print(f"Cohen's h: {cohen_h}")
print(f"Confidence Interval: {confidence_interval}")
print(f"Decision: {decision}")

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(['Control', 'Treatment'], [control_count, treatment_count])
plt.ylabel('Count')
plt.title('Group Counts')
plt.show()