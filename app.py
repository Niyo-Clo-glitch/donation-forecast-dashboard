import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO

# App Configuration
st.set_page_config(page_title="Maison Shalom Dashboard", layout="wide")
st.title("📊 Donation Forecast Dashboard - Maison Shalom")

# Upload CSV
df = None
uploaded_file = st.sidebar.file_uploader("Upload your donation CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check and fix expected columns
    expected_cols = {"Date", "Total_Donations_RWF", "Donor", "Campaign_Type", "Region"}
    if not expected_cols.issubset(df.columns):
        st.error(f"❌ Your file is missing one or more required columns: {expected_cols}")
        st.stop()

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

    # ---------------- Bar Chart Section ----------------
    st.markdown("## 📊 Bar Chart of Donations (Grouped Quarterly)")

    filtered_df['Quarter'] = filtered_df['Date'].dt.to_period("Q").dt.start_time
    grouped = filtered_df.groupby(['Quarter', 'Donor', 'Campaign_Type'])['Total_Donations_RWF'].sum().reset_index()

    try:
        pivot = grouped.pivot_table(index='Quarter', columns=['Donor', 'Campaign_Type'], values='Total_Donations_RWF', aggfunc='sum').fillna(0)
        st.bar_chart(pivot)
    except Exception as e:
        st.error(f"Error generating bar chart: {e}")

    # ---------------- Forecast Section ----------------
    st.markdown("## 🔮 Forecast Total Donations (Next 3 Years)")

    monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Total_Donations_RWF'].sum().reset_index()
    monthly['Date'] = monthly['Date'].dt.to_timestamp()

    forecast_df = monthly.rename(columns={"Date": "ds", "Total_Donations_RWF": "y"})

    if len(forecast_df) < 2:
        st.warning("⚠️ Not enough data to generate a forecast. Please upload a file with more data.")
    else:
        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=36, freq='M')  # 3 years
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # ---------------- Download Section ----------------
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
