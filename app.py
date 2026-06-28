import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# 1. Download Data - Using 'auto_adjust=True' to ensure a cleaner structure
df = yf.download('AAPL', start='2012-01-01', end=datetime.now(), auto_adjust=True)

# --- NEW FIX: Flatten the columns if they are MultiIndex ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Filter only the 'Close' column
# We check if 'Close' exists, otherwise use 'Close' from the flattened headers
data = df[['Close']]
dataset = data.values

# 3. Calculate training length
training_data_len = int(np.ceil( len(dataset) * .95 ))

# 4. Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(f"Success! Data shape: {dataset.shape}")
print(f"Total rows: {len(dataset)}")
print(f"Training rows: {training_data_len}")
# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

# We create a loop to build the 60-day windows
for i in range(60, len(train_data)):
    # x_train will contain the 60 previous values
    x_train.append(train_data[i - 60:i, 0])
    # y_train will contain the 61st value (the one we want to predict)
    y_train.append(train_data[i, 0])

    # Just to show you what the first window looks like
    if i <= 61:
        print(f"Window {i - 60} created successfully.")

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
# LSTM expects the input to be 3D: [number of samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print(f"Final x_train shape: {x_train.shape}")
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. Build the LSTM model architecture
model = Sequential()

# First LSTM layer with 128 neurons.
# return_sequences=True because we are adding another LSTM layer after this one.
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# Second LSTM layer with 64 neurons.
# return_sequences=False because we are moving toward the final output.
model.add(LSTM(64, return_sequences=False))

# Dense layers (Fully connected layers) to refine the data
model.add(Dense(25))
model.add(Dense(1)) # The final output: a single predicted price

# 2. Compile the model
# 'adam' is an optimizer that adjusts the weights to reduce error.
# 'mean_squared_error' is the loss function measuring how far off our predictions are.
model.compile(optimizer='adam', loss='mean_squared_error')

# 3. Train the model
# epochs=1: The model will see the entire dataset once.
# batch_size=1: The model updates its weights after every single sample.
print("Training started... this might take a minute.")
model.fit(x_train, y_train, batch_size=1, epochs=1)
print("Training complete!")
# 1. Create the testing data set
# Create a new array containing scaled values from the end of the training set to the end of the data
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]  # These are the actual prices we want to predict

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# 2. Convert the data to a numpy array
x_test = np.array(x_test)

# 3. Reshape the data to 3D for the LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 4. Get the models predicted price values
predictions = model.predict(x_test)

# 5. Inverse transform the predictions
# We need to un-scale the numbers to get the actual Dollar ($) values
predictions = scaler.inverse_transform(predictions)

# 6. Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"The Root Mean Squared Error (RMSE) is: {rmse}")
import matplotlib.pyplot as plt

# 1. Prepare the data for plotting
# 'train' contains the historical data the model saw
train = data[:training_data_len]
# 'valid' contains the actual prices the model tried to guess
valid = data[training_data_len:]
# Add our AI's predictions to the validation dataframe
valid.loc[:, 'Predictions'] = predictions

# 2. Visualize the data
plt.figure(figsize=(16,8))
plt.title('Apple Stock Price Prediction Model - LSTM', fontsize=18)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Close Price USD ($)', fontsize=15)

# Plot the training data
plt.plot(train['Close'], color='blue', alpha=0.5)
# Plot the actual price
plt.plot(valid['Close'], color='orange')
# Plot the predicted price
plt.plot(valid['Predictions'], color='green', linestyle='--')

plt.legend(['Historical Training Data', 'Actual Price (True)', 'AI Prediction'], loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

# 3. Show the actual vs predicted numbers
print("\nRecent Actual vs Predicted Prices:")
print(valid.tail(10))