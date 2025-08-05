import streamlit as st
import pandas as pd
import altair as alt
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ──────────────────────────────────────────────────────────────────────────────
# Page config
st.set_page_config(page_title="Donation Forecast & Analytics Dashboard", layout="wide")

st.title("📊 Donation Forecast & Analytics Dashboard")
# ──────────────────────────────────────────────────────────────────────────────

# 1) Upload CSV
uploaded_file = st.file_uploader("Upload your donations CSV", type="csv")
if uploaded_file is None:
    st.info("👆 Please upload a CSV file to start.")
    st.stop()

# 2) Load & normalize
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower()

# 3) Validate required columns
required = ['date', 'donor', 'campaign_type', 'region', 'total_donations_rwf']
if not all(c in df.columns for c in required):
    st.error(f"CSV must include columns: {', '.join(required)}")
    st.stop()

# 4) Parse dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])  # drop rows with invalid dates

# 5) Sidebar filters
st.sidebar.header("🔍 Filters")
donors   = st.sidebar.multiselect("Select Donor(s)", options=df['donor'].unique(), default=df['donor'].unique())
types    = st.sidebar.multiselect("Select Campaign Type(s)", options=df['campaign_type'].unique(), default=df['campaign_type'].unique())
regions  = st.sidebar.multiselect("Select Region(s)", options=df['region'].unique(), default=df['region'].unique())

# 6) Apply filters
df_filt = df[
    df['donor'].isin(donors) &
    df['campaign_type'].isin(types) &
    df['region'].isin(regions)
]

# ──────────────────────────────────────────────────────────────────────────────
# TIME SERIES
st.subheader("📈 Total Donations Over Time")
time_group = (
    df_filt
    .groupby(pd.Grouper(key='date', freq='M'))
    .sum(numeric_only=True)
    .reset_index()
)
chart = alt.Chart(time_group).mark_line(point=True).encode(
    x=alt.X('date:T', title='Month'),
    y=alt.Y('total_donations_rwf:Q', title='Total Donations (RWF)'),
    tooltip=['date', 'total_donations_rwf']
).properties(width=800, height=400)
st.altair_chart(chart, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY METRICS
st.subheader("📊 Summary Statistics")
total_amt = int(df_filt['total_donations_rwf'].sum())
total_rec = df_filt.shape[0]
col1, col2 = st.columns(2)
col1.metric("Total Donations (RWF)", f"{total_amt:,.0f}")
col2.metric("Total Records", f"{total_rec}")

# ──────────────────────────────────────────────────────────────────────────────
# BAR CHART BY DONOR
st.subheader("📊 Donations by Donor")
donor_group = (
    df_filt
    .groupby('donor')['total_donations_rwf']
    .sum()
    .reset_index()
    .sort_values('total_donations_rwf', ascending=False)
)
chart2 = alt.Chart(donor_group).mark_bar().encode(
    x=alt.X('donor', sort='-y', title='Donor'),
    y=alt.Y('total_donations_rwf:Q', title='Total Donations (RWF)'),
    tooltip=['donor', 'total_donations_rwf']
).properties(width=800, height=400)
st.altair_chart(chart2, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# FORECASTING: Semi-Annual & Annual, Past 5 Years → Next 5 Years
st.subheader("🔮 Forecast: Past 5 Years & Next 5 Years")

# 1) Restrict to last 5 years
now = datetime.now()
cutoff = now - relativedelta(years=5)
df5 = df_filt[df_filt['date'] >= cutoff]

# -- Semi-Annual (6M)
semi = (
    df5
    .set_index('date')
    .resample('6M')['total_donations_rwf']
    .sum()
    .reset_index()
)
semi.columns = ['ds', 'y']

if len(semi) >= 2:
    st.markdown("**Semi-Annual (6M) Forecast**")
    m_semi = Prophet()
    m_semi.fit(semi)
    future_semi = m_semi.make_future_dataframe(periods=10, freq='6M')
    fc_semi = m_semi.predict(future_semi)
    st.plotly_chart(plot_plotly(m_semi, fc_semi), use_container_width=True)
else:
    st.warning("Not enough semi-annual data points to forecast (need ≥2).")

# -- Annual (1Y)
annual = (
    df5
    .set_index('date')
    .resample('Y')['total_donations_rwf']
    .sum()
    .reset_index()
)
annual.columns = ['ds', 'y']

if len(annual) >= 2:
    st.markdown("**Annual (1Y) Forecast**")
    m_ann = Prophet()
    m_ann.fit(annual)
    future_ann = m_ann.make_future_dataframe(periods=5, freq='Y')
    fc_ann = m_ann.predict(future_ann)
    st.plotly_chart(plot_plotly(m_ann, fc_ann), use_container_width=True)
else:
    st.warning("Not enough annual data points to forecast (need ≥2).")
