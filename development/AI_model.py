import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib  # For saving models and scalers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load and preprocess data
def preprocess_data(data):
    # Example: Normalize and segment data
    return data

data = pd.read_csv('fall_detection_dataset_realistic.csv')  # Load your dataset
X = preprocess_data(data[['gyro_x', 'gyro_y', 'gyro_z']])
y = data['label']  # Labels: 'fall' or 'no fall'

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Example with a Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Predict and evaluate
predictions = rf_model.predict(X_test_scaled)
print(classification_report(y_test, predictions))

# Example with an LSTM model
X_train_lstm = np.expand_dims(X_train_scaled, axis=2)
X_test_lstm = np.expand_dims(X_test_scaled, axis=2)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_lstm, y_train, epochs=10, verbose=1)

# Save the LSTM model
model.save('lstm_model.h5')

# Plot accuracy and loss during training
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

# Evaluate LSTM model
loss, accuracy = model.evaluate(X_test_lstm, y_test, verbose=0)
print(f'Accuracy: {accuracy:.4f}')
