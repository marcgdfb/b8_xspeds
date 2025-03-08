import numpy as np
import matplotlib.pyplot as plt

# Number of samples
N = 100000

# 1. Incorrect: Uniform theta sampling
theta_uniform = np.random.uniform(0, np.pi, N)

# 2. Incorrect: Uniform sin(theta) sampling
sin_theta_uniform = np.random.uniform(0, 1, N)
theta_sin_uniform = np.arcsin(sin_theta_uniform)  # arcsin to get theta

# 3. Correct: Uniform cos(theta) sampling
cos_theta_uniform = np.random.uniform(-1, 1, N)
theta_cos_uniform = np.arccos(cos_theta_uniform)  # arccos to get theta

# Plot histograms
plt.figure(figsize=(12, 5))

bins = np.linspace(0, np.pi, 50)

plt.hist(theta_uniform, bins, density=True, alpha=0.6, label="Uniform θ (Incorrect)")
plt.hist(theta_sin_uniform, bins, density=True, alpha=0.6, label="Uniform sin(θ) (Incorrect)")
plt.hist(theta_cos_uniform, bins, density=True, alpha=0.6, label="Uniform cos(θ) (Correct)")

plt.xlabel("θ (radians)")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Comparing Sampling Methods for θ")
plt.show()
