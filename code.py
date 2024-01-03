pip install tensorflow numpy pandas
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read data from CSV file
df = pd.read_csv('your_data_file.csv')  # Replace 'your_data_file.csv' with the path to your CSV file

# Convert data to NumPy arrays
data = df.values
inputs = data[:, :2]
targets = data[:, 2]

# Create a sequential model
model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),  # First hidden layer with 10 neurons
    Dense(10, activation='relu'),                    # Second hidden layer with 10 neurons
    Dense(1)                                         # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(inputs, targets, epochs=100, verbose=1)

# Evaluate the model (optional)
loss = model.evaluate(inputs, targets, verbose=0)
print(f'Loss (MSE): {loss}')

# Predict using the trained model (optional)
predictions = model.predict(inputs)
print('Predictions:')
print(predictions)
