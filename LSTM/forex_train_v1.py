import psycopg2
from datetime import datetime, timedelta
import threading
import schedule
import subprocess
import time
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import MetaTrader5 as mt5
import pytz
import psycopg2
import logging
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


os.environ["SM_FRAMEWORK"] = "tf.keras"


def get_db_conn():
    conn = psycopg2.connect(database="forex_db",
                             user='postgres', 
                             password='admin', 
                             host='127.0.0.1', 
                             port='5432')

    return conn


def get_data(currency):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        current_datetime = datetime.now()

        postgres_select_query = "SELECT current_datetime, ask_value, bid_value FROM live_data_forex_tbl WHERE currency=%s ORDER BY current_datetime ASC"
        cursor.execute(postgres_select_query, (currency,))
        rows = cursor.fetchall()

        datetime_values = [row[0] for row in rows]
        ask_values = [row[1] for row in rows]
        bid_values = [row[2] for row in rows]

        return datetime_values, ask_values, bid_values
    except Exception as e:
        print("Error:", str(e))
        return None


datetime_, ask, bid = get_data("GBPUSD")
data = pd.DataFrame({'datetime': datetime_, 'ask': ask, 'bid': bid})
data = data.drop(['ask'], axis=1)
print(data)
train_dates = pd.to_datetime(data['datetime'])

data = data.dropna()  # Remove rows with missing values

# Normalize the bid values to be in the range [0, 1]
scaler = MinMaxScaler()
data['bid'] = scaler.fit_transform(data['bid'].values.reshape(-1, 1))

# Step 2: Data Sequencing
# Convert time series data into sequences of fixed length
sequence_length = 48  # Number of past ticks to use as input
future_steps = 1  # Number of future steps to predict
sequences = []
targets = []

for i in range(len(data) - sequence_length - future_steps + 1):
    sequence = data['bid'].values[i:i + sequence_length]
    target = data['bid'].values[i + sequence_length + future_steps - 1]
    sequences.append(sequence)
    targets.append(target)

X = np.array(sequences)
y = np.array(targets)

X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 3: Train-Test Split
# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("X_test", X_test.shape)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))  # First LSTM layer
model.add(LSTM(units=50))  # Second LSTM layer
model.add(Dense(units=1))

# Step 5: Model Training
model.compile(optimizer='adam', loss='mean_squared_error')

# ModelCheckpoint to save the best model based on validation loss
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# EarlyStopping to stop training if the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[checkpoint, early_stopping])

# Step 8: Model Evaluation (Optional)
test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

# Step 7: Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

print(len(predictions))

y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compare predictions to actual bid values
for i in range(len(predictions)):
    print(f"Predicted: {predictions[i][0]}, Actual: {y_test_original[i][0]}")

print(list(train_dates)[-1])

n_future = 300
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future,
                                      freq='S').tolist()  # H - Hour, m - min
print(forecast_period_dates)
