import streamlit as st
from yfinance import download as get
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error
import datetime
import time
import os
import yfinance as yf

st.set_page_config(
    page_title="Cryptocurrency Predictor",
    page_icon='https://clipground.com/images/cryptocurrency-icons-clipart.png',
    layout="centered"
)

def process_data(data, time_period):
    x_data, y_data = [], []
    for i in range(time_period, data.shape[0]):
        x_data.append(data[i - time_period:i, 0])
        y_data.append(data[i, 0])
    return np.array(x_data), np.array(y_data)

def load_models():
    models = []
    for i in range(1, 5):
        model_path = f'models/model{i}.h5'
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                models.append(model)
            except Exception as e:
                st.warning(f"Model {i} could not be loaded due to error: {e}. Skipping...")
        else:
            st.warning(f"Model {i} file not found. Skipping...")
    return models

def fetch_data(symbol):
    """Fetch cryptocurrency data using yfinance."""
    try:
        raw_data = yf.download(f"{symbol}-USD", period="5y", interval="1d")
        return raw_data
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return None

def predict_future(no_days, model, data, scaler):
    x_data, _ = process_data(scaler.transform(data), 30)
    current_input = np.array([x_data[-1]])
    predictions = []
    
    for _ in range(no_days):
        prediction = model.predict(current_input)
        predictions.append(prediction[0][0])
        current_input = np.array([np.append(current_input[0][1:], prediction)])
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

if __name__ == "__main__":
    if "start" not in st.session_state:
        st.session_state.start = False

    if not st.session_state.start:
        st.title("Welcome to Cryptocurrency Predictor! 🚀")
        st.markdown(
            """
            This app helps you predict future cryptocurrency prices using advanced AI models.  
            Select *Start Predictor* below to begin exploring predictions for popular cryptocurrencies.
            """
        )
        if st.button("Start Predictor"):
            st.session_state.start = True
        st.stop()

    coins = {
        'Bitcoin': ('BTC', 1),
        'Ethereum': ('ETH', 2),
        'Cardano': ('ADA', 2),
        'Litecoin': ('LTC', 3),
        'Dogecoin': ('DOGE', 3),
        'Ripple': ('XRP', 4),
    }

    st.sidebar.title("Cryptocurrency Predictor")
    st.title("Cryptocurrency Price Predictor")

    coin = st.sidebar.selectbox("Select Cryptocurrency", options=coins.keys())
    no_days = st.sidebar.slider("Select number of days to predict", min_value=1, max_value=365, value=30)

    models = load_models()
    if len(models) == 0:
        st.error("No models loaded. Please ensure models are available.")
        st.stop()

    symbol, model_idx = coins[coin]
    raw_data = fetch_data(symbol)
    if raw_data is None or raw_data.empty:
        st.error(f"No data available for {coin}.")
        st.stop()

    st.subheader(f"{coin} Historical Data")
    st.write(raw_data.tail(10))

    if 'Adj Close' in raw_data.columns:
        close_price = raw_data[['Adj Close']]
    elif 'Close' in raw_data.columns:
        close_price = raw_data[['Close']]
    else:
        st.error(f"Neither 'Adj Close' nor 'Close' column found for {coin}.")
        st.stop()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(close_price)
    predictions = predict_future(no_days, models[model_idx - 1], scaled_data, scaler)

    future_dates = pd.date_range(start=datetime.datetime.now().date(), periods=no_days)
    prediction_df = pd.DataFrame(predictions, index=future_dates, columns=["Predicted Price"])
    
    st.subheader(f"{coin} Future Predictions")
    st.line_chart(prediction_df)

    st.subheader(f"{coin} Prediction Table")
    st.dataframe(prediction_df)

    x_data, y_data = process_data(scaled_data, 30)
    y_predict = models[model_idx - 1].predict(x_data)
    rmse = np.sqrt(mean_squared_error(y_data, y_predict))
    accuracy = r2_score(y_data, y_predict) * 100

    st.metric(label="Prediction Accuracy", value=f"{accuracy:.2f}%", delta=None)
    st.metric(label="RMSE", value=f"{rmse:.4f}")

    if st.sidebar.button("Exit"):
        if st.sidebar.button("Are you sure you want to exit?"):
            st.write("Thank you for using Cryptocurrency Predictor! See you next time. 👋")
            time.sleep(2)
            st.stop()
