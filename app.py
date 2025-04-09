import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import datetime

# Function to train the LSTM model and make predictions
def train_lstm_model(stock_name, start_date, end_date):
    # Download stock data
    df = yf.download(stock_name, start=start_date, end=end_date)

    # Check if data is empty
    if df.empty:
        st.error("No data found for the given stock symbol and date range. Please try a different symbol or adjust the date range.")
        return None, None, None

    # Prepare data for LSTM model
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Ensure there is enough data for creating sequences
    if len(scaled_data) < 60:
        st.error("Not enough historical data to create sequences for training. Please select a longer date range.")
        return df, [], []

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict future prices
    future_predictions = []
    future_dates = []

    # Use the last 60 days from the available data
    last_60_days = scaled_data[-60:]

    for i in range(30):  # Predict next 30 days
        X_future = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        pred_price = model.predict(X_future)
        future_predictions.append(pred_price[0][0])
        last_60_days = np.append(last_60_days, pred_price)[1:]
        future_dates.append(df.index[-1] + datetime.timedelta(days=i+1))

    predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    return df, predicted_prices, future_dates

# Streamlit UI
st.title('Stock Price Prediction App')

# User inputs
stock_name = st.text_input('Enter Stock Symbol', 'AAPL')
start_date = st.date_input('Start Date', datetime.date(2023, 1, 1))
end_date = st.date_input('End Date', datetime.date.today())

if st.button('Predict'):
    df, predicted_prices, future_dates = train_lstm_model(stock_name, start_date, end_date)

    if df is not None and predicted_prices is not None:
        # Display historical stock prices
        st.subheader('Historical Stock Prices')
        st.line_chart(df['Close'])

        # Display future predicted prices
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predicted_prices.flatten()})
        st.subheader('Future Predicted Prices')
        st.write(future_df)

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Actual Price')
        ax.plot(future_df['Date'], future_df['Predicted Close'], label='Predicted Future Price', color='orange')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{stock_name} Stock Price Prediction')
        ax.legend()
        st.pyplot(fig)
