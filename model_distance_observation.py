import numpy as np
import matplotlib.pyplot as plt
from dimensions import *

def compute_observation_from_distance_to_goal(measurement, ideal_distance=goalkeeper_area_length + robot_diameter / 2, alpha=2):
    normalize_distance = abs(measurement - ideal_distance) / ideal_distance
    p_z = np.exp(-alpha * normalize_distance)
    return p_z

best_distance_to_kick = goalkeeper_area_length + robot_diameter / 2

# Generate a range of measurements from 2000 to 4500
measurements = np.arange(best_distance_to_kick, 4501)

# Compute the observations for each measurement
observations = [compute_observation_from_distance_to_goal(m) for m in measurements]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(measurements, observations, label='p_z vs Measurement')
plt.xlim(best_distance_to_kick, 4500)
plt.ylim(0, 1)
plt.xlabel('Measurement')
plt.ylabel('p_z')
plt.title('Observation from Distance to Goal')
plt.legend()
plt.grid(True)
plt.show()
