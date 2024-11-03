import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyESN import ESN

def esn_stock_prediction(ticker_symbol, num_simulations=100):
    """
    Predicts stock prices using an Echo State Network (ESN) and visualizes the results
    with a cone of uncertainty, and a 30-day prediction.
    """
    # Step 1: Data Collection
    try:
        # Fetch data excluding the problematic date
        data = yf.download(ticker_symbol, start='2015-01-01', end='2024-10-31')
    except Exception as e:
        st.write(f"An error occurred while fetching data: {e}")
        return

    # Check if data is available
    if data.empty:
        st.write(f"No data found for ticker symbol '{ticker_symbol}'.")
        return

    # Step 2: Data Preparation
    closing_prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_prices = scaler.fit_transform(closing_prices.reshape(-1, 1))

    # Split the data into training and testing sets
    train_size = int(len(normalized_prices) * 0.8)
    train_data = normalized_prices[:train_size]
    test_data = normalized_prices[train_size:]

    if len(test_data) < 31:
        st.write("Not enough data for testing after splitting.")
        return

    # Prepare input and output sequences
    train_input = train_data[:-1]
    train_output = train_data[1:]

    test_input = test_data[:-1]
    test_output = test_data[1:]

    # Step 3: Model Training
    esn = ESN(
        n_inputs=1,
        n_outputs=1,
        n_reservoir=500,
        sparsity=0.2,
        random_state=42,
        spectral_radius=0.95
    )
    esn.fit(train_input, train_output)

    # Function to run multiple simulations
    def run_multiple_simulations(esn_model, input_data, num_simulations):
        predictions = []
        for _ in range(num_simulations):
            prediction = esn_model.predict(input_data)
            predictions.append(prediction.flatten())
        return np.array(predictions)

    # Step 4: Prediction and Uncertainty Estimation
    predictions = run_multiple_simulations(esn, test_input, num_simulations)
    mean_prediction = np.mean(predictions, axis=0)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)

    # Step 5: Visualization
    actual_prices = scaler.inverse_transform(test_output)
    mean_pred_prices = scaler.inverse_transform(mean_prediction.reshape(-1, 1))
    lower_bound_prices = scaler.inverse_transform(lower_bound.reshape(-1, 1))
    upper_bound_prices = scaler.inverse_transform(upper_bound.reshape(-1, 1))

    # Plotting the cone of uncertainty
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_prices, label='Actual Prices', color='black')
    ax.plot(mean_pred_prices, label='Mean Prediction', color='blue')
    ax.fill_between(
        range(len(mean_pred_prices)),
        lower_bound_prices.flatten(),
        upper_bound_prices.flatten(),
        color='blue',
        alpha=0.3,
        label='95% Confidence Interval'
    )
    ax.set_title(f'ESN Prediction with Cone of Uncertainty for {ticker_symbol}')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Stock Price')
    ax.legend()

    st.pyplot(fig)

    # Step 6: 30-Day Ahead Prediction
    # Predict the next 30 days
    future_input = normalized_prices[-30:]
    future_prediction = esn.predict(future_input)
    future_prediction_prices = scaler.inverse_transform(future_prediction.reshape(-1, 1))

    # Get actual prices for the next 30 days
    actual_future_prices = closing_prices[-30:]

    # Plotting the 30-day prediction vs actual prices
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(range(30), actual_future_prices, label='Actual 30-Day Prices', color='black')
    ax2.plot(range(30), future_prediction_prices, label='Predicted 30-Day Prices', color='red')
    ax2.set_title(f'30-Day Ahead Prediction vs Actual for {ticker_symbol}')
    ax2.set_xlabel('Days Ahead')
    ax2.set_ylabel('Stock Price')
    ax2.legend()

    st.pyplot(fig2)

    # Step 7: Full Prediction vs Actual Price Visualization
    # Plotting the predicted prices alongside actual prices for the entire testing period
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(range(len(actual_prices)), actual_prices, label='Actual Prices', color='black')
    ax3.plot(range(len(mean_pred_prices)), mean_pred_prices, label='Predicted Prices', color='green')
    ax3.set_title(f'Full Predicted vs Actual Prices for {ticker_symbol}')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Stock Price')
    ax3.legend()

    st.pyplot(fig3)

# Streamlit app
def main():
    st.title("ESN Stock Price Prediction with Cone of Uncertainty")

    # User inputs
    ticker_symbol = st.text_input("Enter the stock ticker symbol (e.g., AAPL):", value='AAPL')
    num_simulations = st.number_input(
        "Number of simulations for uncertainty estimation:",
        min_value=10, max_value=500, value=100, step=10
    )

    if st.button("Run Prediction"):
        esn_stock_prediction(ticker_symbol, num_simulations)

if __name__ == "__main__":
    main()