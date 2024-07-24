import numpy as np
import matplotlib.pyplot as plt
from dimensions import *

def compute_observation_from_closest_enemy_distance(measurement, worst_distance=robot_diameter, alpha=0.5):
    if measurement<robot_diameter:
        return 0
    else:
        normalized_distance = abs(measurement - worst_distance) / worst_distance
        interception_likelihood = np.exp(-alpha * normalized_distance)
        p_z = 1 - interception_likelihood
        return p_z

best_distance_to_enemy = robot_diameter

# Generate a range of measurements from 2000 to 4500
measurements = np.arange(best_distance_to_enemy, 4501)

# Compute the observations for each measurement
observations = [compute_observation_from_closest_enemy_distance(m) for m in measurements]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(measurements, observations, label='p_z vs Measurement', color='blue')
plt.xlim(0, 4500)
plt.ylim(0, 1)
plt.xlabel('Measurement')
plt.ylabel('p_z')
plt.title('Observation from Closest Enemy Distance')
plt.legend()
plt.grid(True)
plt.show()
