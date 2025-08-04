import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO

st.set_page_config(page_title="Quarterly Donation Forecast", layout="wide")
st.title("📈 Quarterly Donation Forecast")

uploaded_file = st.file_uploader("Upload your donation CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Ensure column names are standard
        df.columns = df.columns.str.strip()
        if "Date" not in df.columns or "Total_Donations_RWF" not in df.columns:
            st.error("CSV must contain 'Date' and 'Total_Donations_RWF' columns.")
            st.stop()

        df = df[["Date", "Total_Donations_RWF"]]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # Display raw data
        st.subheader("Raw Donation Data")
        st.dataframe(df.tail(10))

        # Resample to quarterly sums
        df_quarterly = df.resample("Q", on="Date").sum().reset_index()

        # Prophet formatting
        prophet_df = df_quarterly.rename(columns={"Date": "ds", "Total_Donations_RWF": "y"})

        # Forecasting for 3 years = 12 quarters
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=12, freq="Q")
        forecast = model.predict(future)

        # Plot forecast
        st.subheader("Forecasted Donations (Next 3 Years)")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Bar chart: Quarterly historical + forecast
        st.subheader("📊 Quarterly Donations Overview")
        bar_df = forecast[["ds", "yhat"]].copy()
        bar_df.set_index("ds", inplace=True)
        bar_df = bar_df.tail(20)  # Show 5 years: last 2 actual + 3 forecast

        fig2, ax = plt.subplots(figsize=(12, 5))
        bar_df["yhat"].plot(kind="bar", ax=ax)
        ax.set_title("Quarterly Donations (Actual + Forecast)")
        ax.set_ylabel("RWF")
        ax.set_xlabel("Quarter")
        st.pyplot(fig2)

        # Download forecast CSV
        st.subheader("📥 Download Forecast Data")
        download_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        download_df = download_df.rename(columns={
            "ds": "Date",
            "yhat": "Forecast",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound"
        })

        buffer = BytesIO()
        download_df.to_csv(buffer, index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=buffer.getvalue(),
            file_name="donation_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
