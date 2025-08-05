import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st
from io import BytesIO

# Streamlit Page Setup
st.set_page_config(page_title="Maison Shalom Dashboard", layout="wide")
st.title("📊 Donation Forecast Dashboard - Maison Shalom")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("📂 Upload Donation CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter Sidebar
    st.sidebar.subheader("🔍 Filter Options")
    selected_donors = st.sidebar.multiselect("Select Donors", df["Donor"].unique(), default=list(df["Donor"].unique()))
    selected_campaigns = st.sidebar.multiselect("Select Campaign Types", df["Campaign_Type"].unique(), default=list(df["Campaign_Type"].unique()))
    selected_regions = st.sidebar.multiselect("Select Regions", df["Region"].unique(), default=list(df["Region"].unique()))

    # Apply filters
    filtered_df = df[
        df["Donor"].isin(selected_donors) &
        df["Campaign_Type"].isin(selected_campaigns) &
        df["Region"].isin(selected_regions)
    ]

    # Show filtered preview
    st.subheader("📄 Filtered Data Preview")
    st.dataframe(filtered_df.head())

    # Bar Chart - Quarterly Grouping
    st.subheader("📊 Quarterly Donations by Donor & Campaign Type")
    filtered_df["Quarter"] = filtered_df["Date"].dt.to_period("Q").dt.start_time
    grouped_quarterly = filtered_df.groupby(["Quarter", "Donor", "Campaign_Type"])["Total_Donations_RWF"].sum().reset_index()
    pivot_quarterly = grouped_quarterly.pivot_table(index="Quarter", columns=["Donor", "Campaign_Type"], values="Total_Donations_RWF", aggfunc="sum").fillna(0)
    st.bar_chart(pivot_quarterly)

    # Forecasting - Next 3 Years
    st.subheader("🔮 Forecast Total Donations (Next 3 Years)")
    monthly_data = filtered_df.groupby(filtered_df["Date"].dt.to_period("M"))["Total_Donations_RWF"].sum().reset_index()
    monthly_data["Date"] = monthly_data["Date"].dt.to_timestamp()

    forecast_df = monthly_data.rename(columns={"Date": "ds", "Total_Donations_RWF": "y"})
    model = Prophet()
    model.fit(forecast_df)
    future = model.make_future_dataframe(periods=36, freq="M")
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Download forecast data
    st.subheader("📥 Download Forecast Data")
    forecast_csv = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Forecast_Donations_RWF"}).to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", data=forecast_csv, file_name="donation_forecast.csv", mime="text/csv")

else:
    st.info("📌 Please upload a valid donation CSV file to begin.")
