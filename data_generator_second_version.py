import numpy as np
import pandas as pd

# Parameters
num_samples = 10000
num_features = 3  # Gyroscope axes: x, y, z
fall_window_size = 1000  # Number of samples in a fall window
activity_window_size = 1000  # Number of samples in a non-fall window


def generate_fall_data(num_samples, window_size):
    """Generate synthetic gyroscope data for falls with more realistic characteristics."""
    data = []
    for _ in range(num_samples // window_size):
        # Pre-fall: Normal movement (mild activity or standing)
        pre_fall_segment = np.random.normal(loc=[0, 0, 0], scale=[0.5, 0.5, 0.5], size=(window_size // 2, num_features))

        # Fall: Sharp spike followed by stillness (impact)
        fall_spike = np.random.normal(loc=[20, 20, 20], scale=[5, 5, 5],
                                      size=(window_size // 10, num_features))  # Short spike
        post_fall_segment = np.random.normal(loc=[0, 0, 0], scale=[0.1, 0.1, 0.1],
                                             size=(window_size // 2 - window_size // 10, num_features))

        # Combine pre-fall, fall, and post-fall data
        fall_segment = np.vstack((pre_fall_segment, fall_spike, post_fall_segment))

        # Label 1 for fall
        labels = np.ones(window_size)

        data.append(np.column_stack((fall_segment, labels)))

    return np.vstack(data)


def generate_activity_data(num_samples, window_size):
    """Generate synthetic gyroscope data for normal activities (more periodic/random)."""
    data = []
    for _ in range(num_samples // window_size):
        # Simulate different activities with periodic movement (like walking)
        t = np.linspace(0, 4 * np.pi, window_size)  # Simulating a periodic activity (e.g., walking)
        activity_segment = np.column_stack([
            np.sin(t) + np.random.normal(0, 0.2, size=window_size),  # Gyro X axis
            np.sin(t + np.pi / 2) + np.random.normal(0, 0.2, size=window_size),  # Gyro Y axis
            np.sin(t + np.pi / 4) + np.random.normal(0, 0.2, size=window_size)  # Gyro Z axis
        ])

        # Add some noise to simulate different activity types
        activity_segment += np.random.normal(0, 0.5, size=activity_segment.shape)

        # Label 0 for normal activity
        labels = np.zeros(window_size)

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
df.to_csv('fall_detection_dataset_realistic.csv', index=False)

print("Realistic dataset created and saved as 'fall_detection_dataset_realistic.csv'.")
