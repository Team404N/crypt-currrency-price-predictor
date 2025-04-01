import os
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Cryptocurrency Predictor",
    page_icon='https://clipground.com/images/cryptocurrency-icons-clipart.png',
)

# Fixed directory path for the plots
PLOT_DIR = r"D:\abhi\Cryptocurrency-Price-Predictor\Cryptocurrency-Price-Predictor\plots"

# List of cryptocurrencies
coins = [
    'Bitcoin', 'Bitcoin SV', 'Bitcoin Cash', 'Ethereum',
    'Ethereum Classic', 'Cardano', 'Litecoin', 'Dogecoin',
    'Bitcoin Gold', 'Ripple', 'Stellar', 'EOS',
    'Binance Coin', 'Huobi Token', 'OKB'
]

# User inputs
st.title("Cryptocurrency Predictor")
st.sidebar.title("User Input")

# Select cryptocurrency
selected_coin = st.sidebar.selectbox("Select a Cryptocurrency", coins)

# Select prediction range
days_to_predict = st.sidebar.slider(
    "Select Prediction Range (days)", min_value=0, max_value=365, value=30
)
st.sidebar.info("Use the slider to set the number of days for predictions.")

# Load historical data simulation
def generate_historical_data(coin, days=365):
    """
    Simulate historical price data for a given cryptocurrency.
    :param coin: Name of the cryptocurrency.
    :param days: Number of days for historical data.
    :return: Pandas DataFrame containing historical prices.
    """
    np.random.seed(hash(coin) % (2**32))  # Seed for reproducibility per coin
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    prices = np.cumsum(np.random.normal(loc=0, scale=5, size=days)) + 100
    return pd.DataFrame({'Date': dates, 'Price': prices})

# Generate historical data
historical_data = generate_historical_data(selected_coin)

# Simulate predictions based on user input
def generate_predictions(data, days):
    """
    Generate prediction data based on historical trends.
    :param data: Historical data DataFrame.
    :param days: Number of days to predict.
    :return: Pandas DataFrame containing predictions.
    """
    last_date = data['Date'].iloc[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    last_price = data['Price'].iloc[-1]
    predicted_prices = last_price + np.cumsum(np.random.normal(loc=0, scale=2, size=days))
    return pd.DataFrame({'Date': prediction_dates, 'Price': predicted_prices})

predictions = generate_predictions(historical_data, days_to_predict)

# Display metrics
st.subheader("Model Evaluation Results")
st.metric("Prediction Days", f"{days_to_predict} days")
st.metric("Last Known Price", f"${historical_data['Price'].iloc[-1]:.2f}")

st.divider()

# Display plots
st.subheader(f"Historical and Predicted Prices for {selected_coin}")

# Combine historical and prediction data
combined_data = pd.concat([historical_data, predictions])

# Plot data
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=historical_data['Date'], y=historical_data['Price'],
    mode='lines', name='Historical Data',
    line=dict(color='blue')
))
if days_to_predict > 0:
    fig.add_trace(go.Scatter(
        x=predictions['Date'], y=predictions['Price'],
        mode='lines', name='Predicted Data',
        line=dict(color='orange', dash='dash')
    ))
fig.update_layout(
    title=f"{selected_coin} Price Trends",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend_title="Data Type",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Fun animation effects
#if st.button("Celebrate Results"):
  #  st.balloons()
