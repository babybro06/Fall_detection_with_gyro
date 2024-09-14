import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, \
    accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Attention, Flatten, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
import optuna


# Load and preprocess data
def preprocess_data(data, window_size=65):
    from scipy.ndimage import uniform_filter1d
    # Apply smoothing
    data[['gyro_x', 'gyro_y', 'gyro_z']] = uniform_filter1d(data[['gyro_x', 'gyro_y', 'gyro_z']], size=3, axis=0)

    # Segment data into rolling windows
    segmented_data, labels = [], []
    for i in range(0, len(data) - window_size, window_size):
        window = data[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i:i + window_size].values
        label = data['label'].iloc[i:i + window_size].mode()[0]  # Get dominant label
        segmented_data.append(window)
        labels.append(label)

    return np.array(segmented_data), np.array(labels)


# Load dataset
data = pd.read_csv('fall_detection_dataset_realistic.csv')

# Preprocess data
X, y = preprocess_data(data, window_size=65)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 3)  # Flatten the 3 features
X_test_flat = X_test.reshape(-1, 3)

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# Save the scaler for future use
import joblib

joblib.dump(scaler, 'scaler4.pkl')


# ---------------- Functional Model with CNN + LSTM + Attention ----------------
def create_model(trial):
    input_layer = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))

    # CNN layer for feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)

    # LSTM layer for sequential data
    lstm_units = trial.suggest_int('units_lstm', 50, 150)
    lstm_out = LSTM(lstm_units, return_sequences=True)(x)
    # Attention mechanism
    attention = Attention()([lstm_out, lstm_out])

    # Combine LSTM output and attention
    combined = Add()([lstm_out, attention])

    # Flatten the output and pass through dense layers
    combined = Flatten()(combined)
    dense_units = trial.suggest_int('units_dense', 50, 100)
    dense = Dense(dense_units, activation='relu')(combined)

    # Dropout for regularization
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    dropout = Dropout(dropout_rate)(dense)

    # Output layer for binary classification
    output_layer = Dense(1, activation='sigmoid')(dropout)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Nadam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# ---------------- Hyperparameter Optimization using Optuna ----------------
def objective(trial):
    model = create_model(trial)
    early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min', verbose=1)

    history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[early_stopping])
    val_loss = history.history['loss'][-1]

    return val_loss


# Use Optuna for hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Train the best model found by Optuna
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Build the model with the best hyperparameters
best_model = create_model(study.best_trial)
history = best_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, verbose=1,
                         callbacks=[EarlyStopping(monitor='loss', patience=3)])

# Save the best model
best_model.save('best_cnn_lstm_attention_model4.h5')

# ---------------------------- Model Evaluation ----------------------------
# Predict on the test set
y_pred = best_model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate Evaluation Metrics
f1 = f1_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
accuracy = accuracy_score(y_test, y_pred_binary)

# Print the calculated metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# Plot accuracy and loss
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
loss, accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
