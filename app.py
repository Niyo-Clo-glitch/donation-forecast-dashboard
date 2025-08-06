import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import io
from fpdf import FPDF

#----------------------------------
# Utility Functions with Caching
#----------------------------------
@st.cache_data
def load_data(uploaded_file):
    """
    Load and preprocess donation data:
      - auto-detect date column
      - parse dates
      - rename for Prophet
    """
    # Read full data
    df = pd.read_csv(uploaded_file)

    # Auto-detect date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        st.error("No date-like column found. Please ensure one column contains 'date'.")
        return None
    date_col = date_cols[0]

    # Convert to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        st.error(f"Failed to parse dates in column '{date_col}'.")
        return None

    # Ensure donation column exists
    y_cols = [c for c in df.columns if "donation" in c.lower() and "rwf" in c.lower()]
    if not y_cols:
        st.error("No 'total_donations_rwf' column found.")
        return None
    y_col = y_cols[0]

    # Prepare for Prophet
    df = df.rename(columns={date_col: "ds", y_col: "y"})[["ds", "y", *[c for c in df.columns if c not in [date_col, y_col]]]]
    return df


def filter_data(df, donors, campaigns, regions):
    """
    Apply sidebar filters
    """
    if donors:
        df = df[df["donor"].isin(donors)]
    if campaigns:
        df = df[df["campaign_type"].isin(campaigns)]
    if regions:
        df = df[df["region"].isin(regions)]
    return df

#----------------------------------
# PDF Report Builder
#----------------------------------
def create_pdf_report(figs, titles):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for fig, title in zip(figs, titles):
        img_bytes = fig.to_image(format="png")
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.image(io.BytesIO(img_bytes), x=10, y=30, w=pdf.w - 20)
    return pdf.output(dest="S").encode("latin-1")

#----------------------------------
# Main App
#----------------------------------
def main():
    st.set_page_config(page_title="Donation Forecast & Analytics", layout="wide")
    st.title("📊 Donation Forecast & Analytics Dashboard")

    # Sidebar
    st.sidebar.header("Data & Filters")
    uploaded = st.sidebar.file_uploader("Upload your donations CSV", type=["csv"])
    if not uploaded:
        st.info("Awaiting CSV upload…")
        return

    df = load_data(uploaded)
    if df is None:
        return

    # Filters
    st.sidebar.subheader("Donors")
    donors = st.sidebar.multiselect("Select donors", options=df.get("donor", pd.Series()).dropna().unique())
    st.sidebar.subheader("Campaign Types")
    campaigns = st.sidebar.multiselect("Select campaign types", options=df.get("campaign_type", pd.Series()).dropna().unique())
    st.sidebar.subheader("Regions")
    regions = st.sidebar.multiselect("Select regions", options=df.get("region", pd.Series()).dropna().unique())

    filtered = filter_data(df, donors, campaigns, regions)

    # Metrics
    st.subheader("Key Metrics")
    total = filtered["y"].sum()
    count = filtered.shape[0]
    avg = filtered["y"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Donations (RWF)", f"{total:,.0f}")
    c2.metric("Record Count", f"{count}")
    c3.metric("Avg Donation (RWF)", f"{avg:,.0f}")

    # Charts
    st.subheader("Total Donations Over Time")
    fig1 = px.line(filtered, x="ds", y="y", labels={"ds": "Date", "y": "Donations (RWF)"})
    st.plotly_chart(fig1, use_container_width=True)
    buf1 = fig1.to_image(format="png")
    st.download_button("Download Time Series Chart", buf1, "donations_time.png", "image/png")

    st.subheader("Donations by Donor")
    if "donor" in filtered.columns:
        donor_sum = filtered.groupby("donor")["y"].sum().reset_index()
        fig2 = px.bar(donor_sum, x="donor", y="y", labels={"donor": "Donor", "y": "Donations (RWF)"})
        st.plotly_chart(fig2, use_container_width=True)
        buf2 = fig2.to_image(format="png")
        st.download_button("Download Donor Chart", buf2, "donations_by_donor.png", "image/png")

    # Forecast
    st.sidebar.subheader("Forecast Settings")
    periods = st.sidebar.slider("Forecast horizon (months)", 1, 60, 12)
    if st.sidebar.button("Run Forecast"):
        with st.spinner("Training model…"):
            model = Prophet()
            model.fit(filtered[["ds", "y"]])
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)

        st.subheader(f"{periods}-Month Forecast")
        fig3 = px.line(forecast, x="ds", y="yhat", labels={"ds": "Date", "yhat": "Forecast (RWF)"})
        st.plotly_chart(fig3, use_container_width=True)
        buf3 = fig3.to_image(format="png")
        st.download_button("Download Forecast Chart", buf3, "donations_forecast.png", "image/png")

        # PDF report
        report = create_pdf_report([fig1, fig2 if 'fig2' in locals() else fig1, fig3], ["Donations Over Time", "Donor Breakdown", f"{periods}-Month Forecast"])
        st.sidebar.download_button("Download Full PDF Report", report, "donation_report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
