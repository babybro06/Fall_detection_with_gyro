import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
def preprocess_data(data):
    # Example: Normalize and segment data
    return data

data = pd.read_csv('fall_detection_dataset.csv')  # Load your dataset
X = preprocess_data(data[['gyro_x', 'gyro_y', 'gyro_z']])
y = data['label']  # Labels: 'fall' or 'no fall'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example with a Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
print(classification_report(y_test, predictions))

# Example with an LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy:.4f}')
