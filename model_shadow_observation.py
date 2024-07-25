import numpy as np
import matplotlib.pyplot as plt
from dimensions import *

def compute_observation_from_free_space_on_goal(measurement, ideal_space=goal_width, alpha=1):
    normalize_space = abs(measurement - ideal_space) / ideal_space
    p_z = np.exp(-alpha * normalize_space)
    return p_z

# Generate a range of measurements from 2000 to 4500
measurements = np.arange(0, goal_width)

# Compute the observations for each measurement
observations = [compute_observation_from_free_space_on_goal(m) for m in measurements]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(measurements, observations, label='p_z vs Measurement', color='blue')
plt.xlim(0, goal_width)
plt.ylim(0, 1)
plt.xlabel('Measurement')
plt.ylabel('p_z')
plt.title('Observation from Closest Enemy Distance')
plt.legend()
plt.grid(True)
plt.show()
