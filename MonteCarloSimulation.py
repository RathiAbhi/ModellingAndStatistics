import numpy as np
import matplotlib.pyplot as plt

# 📌 Number of Random Points
N = 10000

# 📌 Generate Random (x, y) Points in the Range [-1,1]
x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)

# 📌 Compute Distance from Origin
distance = np.sqrt(x**2 + y**2)

# 📌 Count Points Inside the Circle (r <= 1)
inside_circle = distance <= 1

# 📌 Estimate Pi
pi_estimate = 4 * np.sum(inside_circle) / N

print(f"Estimated Value of Pi: {pi_estimate}")

# 📌 Visualization
plt.figure(figsize=(6,6))
plt.scatter(x[inside_circle], y[inside_circle], color="blue", s=1, label="Inside Circle")
plt.scatter(x[~inside_circle], y[~inside_circle], color="red", s=1, label="Outside Circle")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.legend()
plt.title("Monte Carlo Simulation for Estimating Pi")
plt.show(block=True)