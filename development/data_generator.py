import numpy as np
import pandas as pd

# Parameters
num_samples = 10000
num_features = 3  # Gyroscope axes: x, y, z
fall_window_size = 100  # Number of samples in a fall window
activity_window_size = 100  # Number of samples in a non-fall window

def generate_fall_data(num_samples, window_size):
    """Generate synthetic gyroscope data for falls."""
    data = []
    for _ in range(num_samples // window_size):
        fall_segment = np.random.normal(loc=[0, 0, 0], scale=[1, 1, 1], size=(window_size, num_features))
        fall_segment[window_size//2:] = np.random.normal(loc=[0, 0, 0], scale=[10, 10, 10], size=(window_size//2, num_features))
        labels = np.ones(window_size)  # Label 1 for falls
        data.append(np.column_stack((fall_segment, labels)))
    return np.vstack(data) 

def generate_activity_data(num_samples, window_size):
    """Generate synthetic gyroscope data for normal activities."""
    data = []
    for _ in range(num_samples // window_size):
        activity_segment = np.random.normal(loc=[0, 0, 0], scale=[0.5, 0.5, 0.5], size=(window_size, num_features))
        labels = np.zeros(window_size)  # Label 0 for no falls
        data.append(np.column_stack((activity_segment, labels)))
    return np.vstack(data)

# Generate data
fall_data = generate_fall_data(num_samples, fall_window_size)
activity_data = generate_activity_data(num_samples, activity_window_size)

# Combine and shuffle data
dataset = np.vstack((fall_data, activity_data))
np.random.shuffle(dataset)

# Convert to DataFrame
df = pd.DataFrame(dataset, columns=['gyro_x', 'gyro_y', 'gyro_z', 'label'])

# Save to CSV
df.to_csv('fall_detection_dataset.csv', index=False)

print("dataset created and saved as 'fall_detection_dataset.csv'.")
