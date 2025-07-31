import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from io import StringIO

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Maison Shalom Donations", layout="wide")
st.title("📊 Maison Shalom Donation Dashboard")

# -------------------- SIDEBAR --------------------
st.sidebar.header("1️⃣ Upload Donation CSV")
uploaded_file = st.sidebar.file_uploader("Upload your donation CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure proper datatypes
    df['Date'] = pd.to_datetime(df['Date'])

    # Filters
    st.sidebar.header("2️⃣ Filter Data")
    donor_list = df['Donor'].unique().tolist()
    campaign_list = df['Campaign_Type'].unique().tolist()
    region_list = df['Region'].unique().tolist()

    selected_donors = st.sidebar.multiselect("Filter by Donor", options=donor_list, default=donor_list)
    selected_campaigns = st.sidebar.multiselect("Filter by Campaign Type", options=campaign_list, default=campaign_list)
    selected_regions = st.sidebar.multiselect("Filter by Region", options=region_list, default=region_list)

    # Filter dataset
    filtered_df = df[
        (df['Donor'].isin(selected_donors)) &
        (df['Campaign_Type'].isin(selected_campaigns)) &
        (df['Region'].isin(selected_regions))
    ]

    # -------------------- RAW DATA --------------------
    st.subheader("📄 Filtered Raw Data")
    st.dataframe(filtered_df)

    # -------------------- BAR CHART --------------------
    st.subheader("📊 Monthly Donations by Donor and Campaign Type")
    monthly_grouped = (
        filtered_df.groupby([pd.Grouper(key='Date', freq='M'), 'Donor', 'Campaign_Type'])['Total_Donations_RWF']
        .sum()
        .reset_index()
    )
    monthly_grouped['Month'] = monthly_grouped['Date'].dt.strftime('%Y-%m')

    plt.figure(figsize=(16, 6))
    sns.barplot(
        data=monthly_grouped,
        x="Month",
        y="Total_Donations_RWF",
        hue="Donor",
        ci=None
    )
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Total Donations (RWF)")
    plt.title("Monthly Donations Grouped by Donor")
    st.pyplot(plt.gcf())
    plt.clf()

    # -------------------- FORECAST --------------------
    st.subheader("🔮 Forecast Total Donations (Next 6 Months)")

    forecast_df = filtered_df.groupby(pd.Grouper(key='Date', freq='M'))['Total_Donations_RWF'].sum().reset_index()
    forecast_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    forecast_chart = model.plot(forecast)
    st.pyplot(forecast_chart)

    # -------------------- DOWNLOAD FORECAST --------------------
    st.subheader("📥 Download Forecasted Data")
    download_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecasted_Donations'})
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name='donation_forecast.csv',
        mime='text/csv'
    )

else:
    st.info("📂 Please upload a CSV file to get started.")

