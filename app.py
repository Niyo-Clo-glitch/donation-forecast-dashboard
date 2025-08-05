import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Maison Shalom Dashboard", layout="wide")
st.title("📊 Donation Forecast Dashboard - Maison Shalom")

# File upload
uploaded_file = st.sidebar.file_uploader("📂 Upload your donation CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    # Filters
    with st.sidebar:
        donors = st.multiselect("Select Donors", options=sorted(df['Donor'].unique()), default=list(df['Donor'].unique()))
        campaigns = st.multiselect("Select Campaign Types", options=sorted(df['Campaign_Type'].unique()), default=list(df['Campaign_Type'].unique()))
        regions = st.multiselect("Select Regions", options=sorted(df['Region'].unique()), default=list(df['Region'].unique()))

    # Apply filters
    filtered_df = df[
        (df['Donor'].isin(donors)) &
        (df['Campaign_Type'].isin(campaigns)) &
        (df['Region'].isin(regions))
    ]

    st.success("✅ File uploaded and data loaded successfully!")
    st.write("### Preview of Uploaded Data")
    st.dataframe(filtered_df.head())

    # Grouped Bar Chart (Quarterly)
    st.markdown("## 📊 Bar Chart of Donations (Grouped Quarterly)")
    filtered_df['Quarter'] = filtered_df['Date'].dt.to_period("Q").dt.start_time
    grouped = filtered_df.groupby(['Quarter', 'Donor', 'Campaign_Type'])['Total_Donations_RWF'].sum().reset_index()
    pivot = grouped.pivot_table(index='Quarter', columns=['Donor', 'Campaign_Type'], values='Total_Donations_RWF').fillna(0)
    st.bar_chart(pivot)

    # Forecasting 5 Years Ahead
    st.markdown("## 🔮 Forecast Total Donations (5 Years Projection)")
    monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Total_Donations_RWF'].sum().reset_index()
    monthly['Date'] = monthly['Date'].dt.to_timestamp()

    forecast_df = monthly.rename(columns={"Date": "ds", "Total_Donations_RWF": "y"})
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=60, freq='M')  # 60 months = 5 years
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Download forecast
    forecast_download = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast_Donations_RWF'})
    csv_data = forecast_download.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", data=csv_data, file_name="forecast_5_years.csv", mime="text/csv")

else:
    st.info("👈 Please upload a CSV file to begin.")
