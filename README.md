
# Fall Detection System using CNN-LSTM with Attention

This project focuses on detecting falls using data from a gyroscope sensor. By combining **Convolutional Neural Networks (CNN)** for feature extraction and **Long Short-Term Memory (LSTM)** networks for handling sequential data, along with an **Attention mechanism**, the system aims to identify fall events with high accuracy.

## Features

- **Data Preprocessing**: Gyroscope sensor data is smoothed and segmented into windows for easier pattern recognition.
- **Model Architecture**: A hybrid model combining CNN, LSTM, and Attention mechanisms to capture spatial and temporal dependencies.
- **Hyperparameter Tuning**: The model uses **Optuna** for efficient hyperparameter optimization.
- **Performance Evaluation**: The system provides evaluation metrics like accuracy, precision, recall, and F1 score, along with confusion matrix visualizations.
- **Graphical User Interface (GUI)**: A simple Tkinter-based interface allows users to input gyroscope data and get predictions on whether a fall occurred or not.

## Requirements

- Python 3.x
- Required Python packages: `tensorflow`, `sklearn`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `optuna`, `tkinter`

Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Project Files

- **AI_model5.py**: Contains the code to preprocess the data, build the CNN-LSTM model, and train it.
- **UI_try_2_code.py**: A Tkinter-based user interface for live predictions using the trained model.
- **fall_detection_dataset_realistic.csv**: The dataset containing gyroscope sensor readings with labeled events (fall or no fall).
- **random_forest_model.pkl / best_cnn_lstm_attention_model.h5**: Pretrained models for fall detection.
- **scaler5.pkl**: StandardScaler object to standardize incoming data for prediction.

## How to Run

1. **Train the model**: You can train the CNN-LSTM-Attention model using the \`fall_detection_model.py\` script. Ensure the dataset is available, and the model will be saved after training.

\`\`\`bash
python AI_model5.py
\`\`\`

2. **Use the GUI for Prediction**: 
   After training the model, you can use the \`UI_try2_code.py\` script to open a user interface where you can input gyroscope data (X, Y, Z values) and predict whether a fall occurred.

\`\`\`bash
python UI_try2_code.py
\`\`\`

3. **Dataset**: The system uses gyroscope data that has been segmented into time windows for model training and testing. Each segment contains sensor data (gyro_x, gyro_y, gyro_z) labeled as either 'Fall' or 'No Fall'.

## Data

- **Dataset Name**: \`fall_detection_dataset_realistic.csv\`
- **Origin**: The dataset was collected from gyroscope sensors during simulated falls and normal activities.
- **Classes**: Binary classification - \`Fall\` (1) or \`No Fall\` (0).

## Model Architecture

The model is built using the following components:
- **CNN**: Extracts relevant features from the gyroscope data.
- **LSTM**: Captures sequential information to understand temporal relationships between movements.
- **Attention**: Enhances important temporal parts of the sequence.
- **Dropout**: Prevents overfitting during training.

## Evaluation

The model is evaluated on both training and testing datasets. The following metrics are reported:
- **Accuracy**: How often the model predicts correctly.
- **Precision**: How accurate fall predictions are.
- **Recall**: How well the model identifies all fall events.
- **F1 Score**: A balance between precision and recall.

## Results

- **Test Accuracy**: ~xx%
- **Confusion Matrix**: Plots the actual vs predicted outcomes.
- **Training Loss and Accuracy Plots**: Visualize how the model improved over time during training.

## Issues Encountered

Some challenges faced include a high false-negative rate, where actual falls were predicted as non-falls. To mitigate this, hyperparameter tuning and advanced model architectures (CNN-LSTM with Attention) were employed. There is still room for improving performance on edge cases.

## Potential Applications

- **Healthcare**: Real-time monitoring of elderly or at-risk individuals.
- **Extreme Sports**: Monitoring falls or accidents during activities like skiing or skateboarding.
- **Workplace Safety**: Detection of accidents in industrial environments.

## Future Work

- Integration with real-time microcontrollers (e.g., Raspberry Pi) for live fall detection.
- Improve model generalization by incorporating a larger and more diverse dataset.


## Authors

- Vedant Pandey

