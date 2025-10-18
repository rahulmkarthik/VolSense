import streamlit as st
import pandas as pd
from volsense_inference.forecast_engine import Forecast

# --- Sidebar ---
st.sidebar.title("⚡ VolSense: Volatility Forecasting Dashboard")
model_version = st.sidebar.text_input("Model version", value="v3")
tickers = st.sidebar.text_input("Enter tickers (comma-separated)", value="AAPL, MSFT")
submit = st.sidebar.button("Run Forecast")

# --- Main ---
st.title("📈 Explainable Volatility Forecasting")
st.markdown("Use the sidebar to select tickers and generate forecasts using the trained VolSense model.")

if submit:
    try:
        tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        fcast = Forecast(model_version=model_version, checkpoints_dir="models")
        preds = fcast.run(tickers_list)

        st.success("✅ Forecast complete!")

        # Display predictions
        st.dataframe(preds)

        # Plot first ticker
        ticker_choice = st.selectbox("Select a ticker to visualize", tickers_list)
        fcast.plot(ticker_choice)
        st.pyplot()

    except Exception as e:
        st.error(f"Error running forecast: {e}")