import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load and preprocess data
def preprocess_data(data, window_size=50):
    """
    Preprocess the gyroscope data by:
    - Applying a smoothing filter
    - Standardizing features
    - Segmenting the data into windows
    """
    # 1. Smooth the data using a uniform filter (moving average)
    data[['gyro_x', 'gyro_y', 'gyro_z']] = uniform_filter1d(data[['gyro_x', 'gyro_y', 'gyro_z']], size=3, axis=0)

    # 2. Segment data into rolling windows
    segmented_data = []
    labels = []

    for i in range(0, len(data) - window_size, window_size):
        window = data[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i:i + window_size].values
        label = data['label'].iloc[i:i + window_size].mode()[0]  # Get the dominant label for the window
        segmented_data.append(window)
        labels.append(label)

    return np.array(segmented_data), np.array(labels)

# Load your dataset
data = pd.read_csv('fall_detection_dataset_realistic.csv')

# Preprocess data
X, y = preprocess_data(data, window_size=50)  # Example window size is 50

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()

# Reshape for scaling (flatten and scale, then reshape back)
X_train_flat = X_train.reshape(-1, 3)  # Flatten the 3 features
X_test_flat = X_test.reshape(-1, 3)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# Save the scaler for future use

joblib.dump(scaler, 'scaler.pkl')
# LSTM Model Definition
model = Sequential([
    LSTM(100, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    Dropout(0.3),  # Prevent overfitting
    Dense(50, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=3)

# Train the LSTM model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[early_stopping])

# Save the LSTM model
model.save('lstm_model.h5')

# Plot accuracy and loss
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the LSTM model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Accuracy: {accuracy:.4f}')