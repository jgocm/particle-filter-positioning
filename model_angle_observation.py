import numpy as np
import matplotlib.pyplot as plt

# Define the function
def compute_observation_from_angle_to_goal(measurement, best_angle, alpha=0.2):
    if measurement<10:
        return 0
    else:
        normalized_angle_diff = abs(measurement - best_angle) / best_angle
        angle_likelihood = np.exp(-alpha * normalized_angle_diff)
        p_z = angle_likelihood
        return p_z

best_angle = 50  # Best angle to goal

# Generate a range of angle measurements from 0 to 180
angle_measurements = np.arange(0, best_angle+1)

# Compute the observations for each measurement
angle_observations = [compute_observation_from_angle_to_goal(m, best_angle) for m in angle_measurements]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(angle_measurements, angle_observations, label='p_z vs Angle Measurement')
plt.xlim(0, best_angle)
plt.ylim(0, 1)
plt.xlabel('Angle Measurement (degrees)')
plt.ylabel('p_z')
plt.title('Observation from Angle to Goal')
plt.legend()
plt.grid(True)
plt.show()
