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
      - parse dates
      - rename columns for Prophet
    """
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    df.rename(columns={"date": "ds", "total_donations_rwf": "y"}, inplace=True)
    return df


def filter_data(df, donors, campaigns, regions):
    """
    Apply sidebar filters to dataframe
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
        # Export Plotly figure to image bytes
        img_bytes = fig.to_image(format="png")
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        # Insert chart image
        pdf.image(io.BytesIO(img_bytes), x=10, y=30, w=pdf.w - 20)
    return pdf.output(dest="S").encode("latin-1")

#----------------------------------
# Main Streamlit App
#----------------------------------
def main():
    st.set_page_config(page_title="Donation Forecast & Analytics", layout="wide")
    st.title("📊 Donation Forecast & Analytics Dashboard")

    # Sidebar: Upload and Filters
    st.sidebar.header("Data & Filters")
    uploaded = st.sidebar.file_uploader("Upload your donations CSV", type=["csv"])
    if not uploaded:
        st.info("Please upload a CSV file to get started.")
        return

    # Load and filter data
    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.sidebar.subheader("Donors")
    donors = st.sidebar.multiselect("Select donors", options=df["donor"].unique())
    st.sidebar.subheader("Campaign Types")
    campaigns = st.sidebar.multiselect("Select campaign types", options=df["campaign_type"].unique())
    st.sidebar.subheader("Regions")
    regions = st.sidebar.multiselect("Select regions", options=df["region"].unique())

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

    # Time Series Chart
    st.subheader("Total Donations Over Time")
    fig1 = px.line(filtered, x="ds", y="y", title="Donations Over Time",
                   labels={"ds": "Date", "y": "Donations (RWF)"})
    st.plotly_chart(fig1, use_container_width=True)
    buf1 = fig1.to_image(format="png")
    st.download_button("Download Time Series Chart", data=buf1,
                       file_name="donations_time.png", mime="image/png")

    # Bar Chart by Donor
    st.subheader("Donations by Donor")
    donor_sum = filtered.groupby("donor")["y"].sum().reset_index()
    fig2 = px.bar(donor_sum, x="donor", y="y", title="Donations by Donor",
                  labels={"donor": "Donor", "y": "Donations (RWF)"})
    st.plotly_chart(fig2, use_container_width=True)
    buf2 = fig2.to_image(format="png")
    st.download_button("Download Donor Bar Chart", data=buf2,
                       file_name="donations_by_donor.png", mime="image/png")

    # Forecast Controls
    st.sidebar.subheader("Forecast Settings")
    periods = st.sidebar.slider("Forecast horizon (months)", 1, 60, 12)

    if st.sidebar.button("Run Forecast"):
        with st.spinner("Training model..."):
            model = Prophet()
            model.fit(filtered[["ds", "y"]])
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)

        # Forecast Chart
        st.subheader(f"{periods}-Month Forecast")
        fig3 = px.line(forecast, x="ds", y="yhat", title="Forecasted Donations",
                       labels={"ds": "Date", "yhat": "Forecast (RWF)"})
        st.plotly_chart(fig3, use_container_width=True)
        buf3 = fig3.to_image(format="png")
        st.download_button("Download Forecast Chart", data=buf3,
                           file_name="donations_forecast.png", mime="image/png")

        # Full PDF Report
        report_bytes = create_pdf_report(
            [fig1, fig2, fig3],
            ["Donations Over Time", "Donations by Donor", f"{periods}-Month Forecast"]
        )
        st.sidebar.download_button(
            "Download Full PDF Report",
            data=report_bytes,
            file_name="donation_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
