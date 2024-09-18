import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
#import tensorflow_addons as tfa


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

# Function to augment data by adding noise
def augment_data(X):
    noise = np.random.normal(0, 0.05, X.shape)
    return X + noise

# Load your dataset
data = pd.read_csv('fall_detection_dataset_realistic.csv')

# Preprocess data
X, y = preprocess_data(data, window_size=85)  # Example window size is 50

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()

# Reshape for scaling (flatten and scale, then reshape back)
X_train_flat = X_train.reshape(-1, 3)  # Flatten the 3 features
X_test_flat = X_test.reshape(-1, 3)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# Augment the training data
X_train_augmented = augment_data(X_train_scaled)

# Save the scaler for future use
joblib.dump(scaler, 'scaler3.pkl')

# LSTM Model Definition with Bidirectional layers and Dropout for regularization
model = Sequential([
    Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(X_train_augmented.shape[1], X_train_augmented.shape[2])),
    Dropout(0.4),  # Higher dropout rate to reduce overfitting
    Bidirectional(LSTM(64, activation='relu')),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0027)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Train the LSTM model
history = model.fit(X_train_augmented, y_train, validation_split=0.2, epochs=30, batch_size=64, callbacks=[early_stopping, reduce_lr])

# Save the LSTM model
model.save('lstm_model3.h5')

# Plot accuracy and loss
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the LSTM model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Accuracy: {accuracy:.4f}')

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)

# Calculate classification metrics
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['No Fall', 'Fall']))

# F1 Score, Precision, Recall, and Accuracy
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fall', 'Fall'], yticklabels=['No Fall', 'Fall'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
