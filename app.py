import streamlit as st
import pandas as pd
import altair as alt
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

st.set_page_config(page_title="Donation Dashboard", layout="wide")

st.title("📊 Donation Forecast & Analytics Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your donations CSV", type="csv")

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check required columns
    required_cols = ['date', 'donor', 'campaign_type', 'region', 'total_donations_rwf']
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must include columns: Date, Donor, Campaign_Type, Region, Total_Donations_RWF")
        st.stop()

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    donors = st.sidebar.multiselect("Select Donor(s)", options=df['donor'].unique(), default=df['donor'].unique())
    campaigns = st.sidebar.multiselect("Select Campaign Type(s)", options=df['campaign_type'].unique(), default=df['campaign_type'].unique())
    regions = st.sidebar.multiselect("Select Region(s)", options=df['region'].unique(), default=df['region'].unique())

    # Filter Data
    df_filtered = df[
        (df['donor'].isin(donors)) &
        (df['campaign_type'].isin(campaigns)) &
        (df['region'].isin(regions))
    ]

    st.subheader("📈 Total Donations Over Time")
    time_group = df_filtered.groupby(pd.Grouper(key='date', freq='M')).sum(numeric_only=True).reset_index()

    chart = alt.Chart(time_group).mark_line(point=True).encode(
        x=alt.X('date:T', title='Month'),
        y=alt.Y('total_donations_rwf:Q', title='Total Donations (RWF)'),
        tooltip=['date', 'total_donations_rwf']
    ).properties(width=800, height=400)

    st.altair_chart(chart)

    # Summary
    st.subheader("📊 Summary Statistics")
    total_donations = int(df_filtered['total_donations_rwf'].sum())
    total_records = df_filtered.shape[0]
    st.metric("Total Donations (RWF)", f"{total_donations:,.0f}")
    st.metric("Total Records", total_records)

    # Breakdown by donor
    st.subheader("📊 Donations by Donor")
    donor_group = df_filtered.groupby('donor')['total_donations_rwf'].sum().reset_index().sort_values(by='total_donations_rwf', ascending=False)
    chart2 = alt.Chart(donor_group).mark_bar().encode(
        x=alt.X('donor', sort='-y'),
        y='total_donations_rwf',
        tooltip=['donor', 'total_donations_rwf']
    ).properties(width=800, height=400)

    st.altair_chart(chart2)

    # Forecasting
    st.subheader("🔮 Forecast Future Donations (Prophet)")

    prophet_data = df_filtered.groupby('date')['total_donations_rwf'].sum().reset_index()
    prophet_data.columns = ['ds', 'y']

    m = Prophet()
    m.fit(prophet_data)

    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig)

else:
    st.info("👆 Please upload a CSV file to start.")

