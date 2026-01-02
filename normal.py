import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

np.random.seed(42)
iq_scores = np.random.normal(100, 15, 1000)

mean = np.mean(iq_scores)
std_dev = np.std(iq_scores)

print(f"Mean IQ Score: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

sns.histplot(iq_scores, bins=30, kde=True, stat="density", color="skyblue")

x = np.linspace(min(iq_scores), max(iq_scores), 200)
y = norm.pdf(x, mean, std_dev)
plt.plot(x, y, color='red', linewidth=2)

plt.title("IQ Score Distribution Using Normal Distribution")
plt.xlabel("IQ Score")
plt.ylabel("Density")
plt.show()

prob_above_130 = 1 - norm.cdf(130, mean, std_dev)
print(f"Probability of IQ > 130: {prob_above_130*100:.2f}%")

prob_90_110 = norm.cdf(110, mean, std_dev) - norm.cdf(90, mean, std_dev)
print(f"Probability of IQ between 90 and 110: {prob_90_110*100:.2f}%")

iq_value = 115
z_score = (iq_value - mean) / std_dev
print(f"Z-score for IQ = {iq_value}: {z_score:.2f}")
