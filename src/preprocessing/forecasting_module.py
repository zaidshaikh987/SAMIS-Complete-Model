# forecasting_module.py

import pandas as pd
import numpy as np
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def preprocess_data(data):
    data = data[['Date', 'Price']]
    data.set_index('Date', inplace=True)
    data = data.resample('D').mean().fillna(method='ffill')
    return data

def prophet_forecasting(data):
    df_prophet = data.reset_index().rename(columns={'Date': 'ds', 'Price': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)  # Forecasting 30 days ahead
    forecast = model.predict(future)
    return model, forecast

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(train_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data)
    
    X, y = [], []
    for i in range(len(scaled_data) - 1):
        X.append(scaled_data[i:i+1])
        y.append(scaled_data[i+1])
    X, y = np.array(X), np.array(y)
    
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=10, verbose=1)
    return model, scaler

def lstm_forecasting(model, scaler, data):
    scaled_data = scaler.transform(data)
    X = np.array([scaled_data[-1:]])
    prediction = model.predict(X)
    return scaler.inverse_transform(prediction)[0][0]

def plot_forecasts(prophet_forecast, lstm_predictions):
    plt.figure(figsize=(14,7))
    plt.subplot(2, 1, 1)
    plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet Forecast')
    plt.plot(prophet_forecast['ds'], prophet_forecast['y'], label='Actual Prices', color='green')
    plt.title('Prophet Forecast')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(lstm_predictions, label='LSTM Predictions', color='red')
    plt.title('LSTM Forecast')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'data/combined_rice_weather_data.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    
    prophet_model, prophet_forecast = prophet_forecasting(data)
    
    train_data = data[['Price']].values
    lstm_model, scaler = train_lstm_model(train_data)
    
    lstm_prediction = lstm_forecasting(lstm_model, scaler, data[['Price']].values)
    
    plot_forecasts(prophet_forecast, [lstm_prediction] * len(data))
    
if __name__ == '__main__':
    main()
