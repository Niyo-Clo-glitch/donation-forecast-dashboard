import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st
from io import BytesIO

# App Configuration
st.set_page_config(page_title="Maison Shalom Dashboard", layout="wide")
st.title("📊 Donation Forecast Dashboard - Maison Shalom")

# Upload CSV
df = None
uploaded_file = st.sidebar.file_uploader("Upload your donation CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    st.success("✅ File uploaded successfully!")
    st.write("### Preview of Uploaded Data", df.head())

    # Filters
    with st.sidebar:
        st.markdown("### Filter Options")
        donors = st.multiselect("Select Donors", options=sorted(df['Donor'].unique()), default=list(df['Donor'].unique()))
        campaigns = st.multiselect("Select Campaign Types", options=sorted(df['Campaign_Type'].unique()), default=list(df['Campaign_Type'].unique()))
        regions = st.multiselect("Select Regions", options=sorted(df['Region'].unique()), default=list(df['Region'].unique()))

    # Apply filters
    filtered_df = df[(df['Donor'].isin(donors)) &
                     (df['Campaign_Type'].isin(campaigns)) &
                     (df['Region'].isin(regions))]

    st.markdown("## 📊 Bar Chart of Donations (Grouped Quarterly)")
    filtered_df['Quarter'] = filtered_df['Date'].dt.to_period("Q").dt.start_time

    grouped = filtered_df.groupby(['Quarter', 'Donor', 'Campaign_Type'])['Amount'].sum().reset_index()

    pivot = grouped.pivot_table(index='Quarter', columns=['Donor', 'Campaign_Type'], values='Amount', aggfunc='sum').fillna(0)
    st.bar_chart(pivot)

    # Forecast Section
    st.markdown("## 🔮 Forecast Total Donations (Next 3 Years)")
    monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
    monthly['Date'] = monthly['Date'].dt.to_timestamp()

    forecast_df = monthly.rename(columns={"Date": "ds", "Amount": "y"})
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=36, freq='M')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Download Section
    st.markdown("### 📥 Download Forecast")
    forecast_to_download = forecast[['ds', 'yhat']].rename(columns={"ds": "Date", "yhat": "Forecast_Donations_RWF"})
    csv_data = forecast_to_download.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Forecast as CSV",
        data=csv_data,
        file_name="donation_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("📂 Please upload a donation CSV file to continue.")
