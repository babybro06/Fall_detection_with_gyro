import sys
import numpy as np
import joblib
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel

# Load the trained model and scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler5.pkl')


class FallDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Fall Detection UI")
        self.setGeometry(100, 100, 300, 200)

        # Create a central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create input fields
        self.gyro_x_entry = QLineEdit()
        self.gyro_x_entry.setPlaceholderText("Gyro X")
        layout.addWidget(self.gyro_x_entry)

        self.gyro_y_entry = QLineEdit()
        self.gyro_y_entry.setPlaceholderText("Gyro Y")
        layout.addWidget(self.gyro_y_entry)

        self.gyro_z_entry = QLineEdit()
        self.gyro_z_entry.setPlaceholderText("Gyro Z")
        layout.addWidget(self.gyro_z_entry)

        # Create a button to make predictions
        predict_button = QPushButton("Predict Fall")
        predict_button.clicked.connect(self.predict_fall)
        layout.addWidget(predict_button)

        # Label to display prediction result
        self.result_label = QLabel("Prediction: ")
        layout.addWidget(self.result_label)

    def preprocess_data(self, data):
        return scaler.transform([data])

    def predict_fall(self):
        try:
            # Get user input from the entry fields
            gyro_x = float(self.gyro_x_entry.text())
            gyro_y = float(self.gyro_y_entry.text())
            gyro_z = float(self.gyro_z_entry.text())

            sensor_data = np.array([gyro_x, gyro_y, gyro_z])
            processed_data = self.preprocess_data(sensor_data)
            prediction = rf_model.predict(processed_data)
            result = 'Fall' if prediction[0] == 1 else 'No Fall'
        except ValueError:
            result = 'Invalid input. Please enter valid numbers.'

        self.result_label.setText(f"Prediction: {result}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FallDetectionApp()
    main_window.show()
    sys.exit(app.exec_())
