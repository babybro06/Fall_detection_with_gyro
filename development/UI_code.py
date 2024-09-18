import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')


def preprocess_data(data):
    return scaler.transform([data])


def predict_fall():
    try:
        # Get user input from the entry fields
        gyro_x = float(gyro_x_entry.get())
        gyro_y = float(gyro_y_entry.get())
        gyro_z = float(gyro_z_entry.get())

        sensor_data = np.array([gyro_x, gyro_y, gyro_z])
        processed_data = preprocess_data(sensor_data)
        prediction = rf_model.predict(processed_data)
        result = 'Fall' if prediction[0] == 1 else 'No Fall'
    except ValueError:
        result = 'Invalid input. Please enter valid numbers.'

    result_label.config(text=f"Prediction: {result}")


# Create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Fall Detection")

    # Create input fields for gyro sensor data
    tk.Label(root, text="Gyro X:").pack(pady=5)
    global gyro_x_entry
    gyro_x_entry = tk.Entry(root)
    gyro_x_entry.pack(pady=5)

    tk.Label(root, text="Gyro Y:").pack(pady=5)
    global gyro_y_entry
    gyro_y_entry = tk.Entry(root)
    gyro_y_entry.pack(pady=5)

    tk.Label(root, text="Gyro Z:").pack(pady=5)
    global gyro_z_entry
    gyro_z_entry = tk.Entry(root)
    gyro_z_entry.pack(pady=5)

    # Add a button to make predictions
    predict_button = ttk.Button(root, text="Predict Fall", command=predict_fall)
    predict_button.pack(pady=20)

    # Label to display prediction result
    global result_label
    result_label = tk.Label(root, text="Prediction: ")
    result_label.pack(pady=20)

    # Run the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    create_gui()
