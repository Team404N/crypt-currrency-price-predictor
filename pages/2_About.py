import streamlit as st
import numpy as np

st.set_page_config(page_title="Cryptocurrency Predictor",page_icon='https://clipground.com/images/cryptocurrency-icons-clipart.png',)
st.markdown(
'''
---
# About
Welcome to the "Cryptocurrency Prediction using LSTM Neural Network" project! This project focuses on utilizing deep learning techniques to forecast cryptocurrency prices by grouping a set of coins that are closely related in terms of market dynamics or share similar price trends.
## Project Objective
The main objective of this project is to develop an accurate and reliable model for predicting cryptocurrency prices using LSTM neural network. By grouping closely related coins, the project aims to capture the temporal dependencies and patterns in the cryptocurrency market data, and use them for making price predictions.
## Project Features
- __Data Collection:__ Historical cryptocurrency price data is collected using the yfinance library, which provides access to financial data from various sources, including cryptocurrency exchanges.
- __Data Preprocessing:__ The collected data is preprocessed using various techniques such as data cleaning, normalization, and feature engineering to prepare it for model training.
- __Model Building:__ LSTM neural network model is built using the Keras library. The model includes layers such as Dense, Dropout, and LSTM to capture the temporal dependencies in the cryptocurrency price data.
- __Grouping of Closely Related Coins:__ Coins that are closely related in terms of market dynamics or share similar price trends are grouped together for more accurate prediction. This approach allows for capturing specific patterns and trends that may be unique to certain groups of coins.
- __Model Training and Evaluation:__ The preprocessed data is split into training and testing sets. The LSTM model is trained on the training set and evaluated on the testing set using metrics such as R2 score and mean squared error (MSE) to assess its performance.
- __Visualization:__ The project includes visualization using the plotly library to visualize the predicted cryptocurrency prices and compare them with the actual prices. This allows for better understanding and interpretation of the model's predictions.
## Project Technologies
The project is implemented in Python 3.x and utilizes the following libraries:
- __yfinance:__ For data collection
- __numpy:__ For numerical computations
- __pandas:__ For data preprocessing
- __scikit-learn:__ For model evaluation
- __Keras:__ For building LSTM neural network model
- __plotly:__ For data visualization
## Project Team
1. __*Name* pin__

## Conclusion
The "Cryptocurrency Prediction using LSTM Neural Network" project aims to provide accurate and reliable cryptocurrency price predictions by leveraging deep learning techniques and grouping closely related coins. The project includes data collection, preprocessing, model building, and evaluation steps, along with visualization and a project website to showcase the results. Thank you for visiting the project website and your interest in our work!
## References
- ChatGPT
- yfinance library documentation
- Keras library documentation
- Scikit-learn library documentation
- Plotly library documentation
- Streamlit
- Numpy
- Pandas
---
''')
# if np.random.choice([True, False], size=1)[0]:
#     st.balloons()
# else:
#     st.snow()