import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import datetime

# Load the saved model
model = load_model('models/model_stockvision.h5')

# App title and subtitle
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #4169E1; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);">
            StockVision: AI enabled Stock Predictions
        </h1>
    </div>
    <div style="text-align: center;">
        <h3>This app is for educational purposes only and not investment advice.</h3>
    </div>
""", unsafe_allow_html=True)

# Sidebar input fields
st.sidebar.header('User Input Parameters')
stock_ticker = st.sidebar.text_input('Stock Ticker', 'GOOGL', help="Use Yahoo Finance Ticker Symbols (e.g., GOOGL for Google).")
prediction_range = st.sidebar.slider('Prediction Range (days)', 1, 30, 7)
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

# Tooltips
st.sidebar.subheader('Tooltips')
st.sidebar.write('''
- **Stock Ticker**: The symbol representing the stock (e.g., GOOGL for Google).
- **Prediction Range**: Number of future days to predict.
- **Start Date**: The start date for fetching historical data.
- **End Date**: The end date for fetching historical data.
- **Predict Button**: Click to generate predictions.
''')

if st.sidebar.button('Predict'):
    # Fetch the data
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    if data.empty:
        st.error('Error fetching data. Please check the stock ticker and date range.')
    else:
        st.subheader(f'Showing data for {stock_ticker} from {start_date} to {end_date}')

        # Prepare data for prediction
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)
        data['Close'] = data['Close'].astype(float)

        close_data = data['Close'].values.reshape(-1, 1)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        # Create a data structure with 60 timesteps and 1 output
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        time_step = 60
        X_data, y_data = create_dataset(scaled_data, time_step)

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

        # Get the predicted stock price
        predictions = model.predict(X_data)
        predictions = scaler.inverse_transform(predictions)

        # Future predictions
        last_60_days = scaled_data[-60:]
        last_60_days = last_60_days.reshape(1, time_step, 1)
        future_predictions = []
        for _ in range(prediction_range):
            predicted_stock_price = model.predict(last_60_days)
            future_predictions.append(predicted_stock_price[0][0])
            last_60_days = np.append(last_60_days[:, 1:, :], predicted_stock_price.reshape(1, 1, 1), axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Generate future dates for plotting
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=prediction_range + 1).tolist()
        future_dates = future_dates[1:]  # remove the first date which is the last known date

        # Plot the actual and predicted prices using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Stock Price'))
        fig.add_trace(go.Scatter(x=data['Date'][time_step + 1:], y=predictions.flatten(), mode='lines', name='Predicted Stock Price'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Future Predictions', line=dict(dash='dash')))

        fig.update_layout(title='Stock Price Prediction and Forecasting',
                          xaxis_title='Date',
                          yaxis_title='Stock Price',
                          template='plotly_white',
                          height=550,  # Increased height
                          width=1800)  # Increased width

        st.plotly_chart(fig)
