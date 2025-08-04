import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st
import numpy as np

# Upload CSV
st.title("Quarterly Donation Forecast")
uploaded_file = st.file_uploader("Upload your donation CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df = df[["Date", "Amount"]]
    
    # Resample quarterly
    df['Quarter'] = df['Date'].dt.to_period('Q')
    quarterly = df.groupby('Quarter').sum().reset_index()
    quarterly['Quarter'] = quarterly['Quarter'].dt.to_timestamp()
    quarterly = quarterly.rename(columns={"Quarter": "ds", "Amount": "y"})

    # Prophet forecasting
    model = Prophet()
    model.fit(quarterly)
    future = model.make_future_dataframe(periods=12, freq='Q')
    forecast = model.predict(future)

    # Forecast plot
    fig = model.plot(forecast)
    st.pyplot(fig)

    # Downloadable forecast
    forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=forecast_out.to_csv(index=False),
        file_name='forecast_3_years.csv',
        mime='text/csv'
    )
