import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.backends.backend_pdf import PdfPages

# ——— Page setup ———
st.set_page_config(page_title="Donation Forecast & Analytics Dashboard", layout="wide")
st.title("Donation Forecast & Analytics Dashboard")

# ——— File uploader ———
uploaded_file = st.file_uploader("Upload your donations CSV", type="csv")
if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# ——— Data load & validation ———
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower()
required = ['date', 'donor', 'campaign_type', 'region', 'total_donations_rwf']
if not all(col in df.columns for col in required):
    st.error(f"CSV must contain: {required}")
    st.stop()

df['date'] = pd.to_datetime(df['date'])

# ——— Sidebar filters ———
st.sidebar.header("Filters")
donors   = st.sidebar.multiselect("Donor", df['donor'].unique(), default=df['donor'].unique())
campaign = st.sidebar.multiselect("Campaign Type", df['campaign_type'].unique(), default=df['campaign_type'].unique())
regions  = st.sidebar.multiselect("Region", df['region'].unique(), default=df['region'].unique())

df_f = df[
    df['donor'].isin(donors) &
    df['campaign_type'].isin(campaign) &
    df['region'].isin(regions)
]

# ——— Chart 1: Total Donations Over Time ———
st.subheader("Total Donations Over Time")
ts = df_f.groupby('date')['total_donations_rwf'].sum().sort_index()
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(ts.index, ts.values, '-o')
ax1.set_xlabel("Date")
ax1.set_ylabel("Total Donations (RWF)")
ax1.set_title("Total Donations Over Time")
st.pyplot(fig1)

# ——— Summary stats ———
st.subheader("Summary Statistics")
total = ts.sum()
count = len(df_f)
col1, col2 = st.columns(2)
col1.metric("Total Donations (RWF)", f"{total:,.0f}")
col2.metric("Total Records", f"{count}")

# ——— Chart 2: Donations by Donor ———
st.subheader("Donations by Donor")
by_donor = df_f.groupby('donor')['total_donations_rwf'].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(by_donor.index, by_donor.values)
ax2.set_xlabel("Donor")
ax2.set_ylabel("Total Donations (RWF)")
ax2.set_title("Donations by Donor")
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig2)

# ——— Chart 3: Forecast Next 3 Years ———
st.subheader("Forecast Next 3 Years (36 Months)")
horizon = 36  # months

prophet_df = df_f[['date', 'total_donations_rwf']].rename(columns={'date': 'ds', 'total_donations_rwf': 'y'})
m = Prophet()
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=horizon, freq='M')
forecast = m.predict(future)

fig3 = m.plot(forecast)
plt.title("Donation Forecast for Next 36 Months")
st.pyplot(fig3)

# ——— Table of Projected Values ———
proj_df = (
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    .tail(horizon)
    .rename(columns={
        'ds': 'Date',
        'yhat': 'Forecast (RWF)',
        'yhat_lower': 'Lower Bound (RWF)',
        'yhat_upper': 'Upper Bound (RWF)'
    })
)
st.subheader("Projected Values (Next 3 Years)")
st.dataframe(proj_df.style.format({
    'Forecast (RWF)': '{:,.0f}',
    'Lower Bound (RWF)': '{:,.0f}',
    'Upper Bound (RWF)': '{:,.0f}'
}), use_container_width=True)

# ——— Download Full Report (PDF) ———
pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer) as pdf:
    # cover page with filters & summary
    fig0, ax0 = plt.subplots(figsize=(8.27, 11))
    ax0.axis('off')
    filter_text = (
        f"Donation Forecast Report\n\n"
        f"Run on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"Filters applied:\n"
        f" • Donors: {', '.join(donors)}\n"
        f" • Campaigns: {', '.join(campaign)}\n"
        f" • Regions: {', '.join(regions)}\n\n"
        f"Summary stats:\n"
        f" • Total records: {count}\n"
        f" • Total donated: {total:,.0f} RWF"
    )
    ax0.text(0.05, 0.95, filter_text, va='top', fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig0)
    plt.close(fig0)

    # chart pages
    pdf.savefig(fig1); plt.close(fig1)
    pdf.savefig(fig2); plt.close(fig2)
    pdf.savefig(fig3); plt.close(fig3)

    # table page
    fig4, ax4 = plt.subplots(figsize=(11, 8))
    ax4.axis('off')
    tbl = ax4.table(
        cellText=[
            [
                row.Date.strftime('%Y-%m'),
                f"{row['Forecast (RWF)']:,.0f}",
                f"{row['Lower Bound (RWF)']:,.0f}",
                f"{row['Upper Bound (RWF)']:,.0f}"
            ]
            for _, row in proj_df.iterrows()
        ],
        colLabels=proj_df.columns,
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    pdf.savefig(fig4)
    plt.close(fig4)

pdf_buffer.seek(0)

st.download_button(
    label="📄 Download Full Report (PDF)",
    data=pdf_buffer,
    file_name="Donation_Forecast_Report_3yr.pdf",
    mime="application/pdf",
)
