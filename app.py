import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.io as pio
import io
from fpdf import FPDF

#----------------------------------
# Utility Functions
#----------------------------------
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        st.error("No date-like column found. Ensure a column name contains 'date'.")
        return None
    date_col = date_cols[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        st.error(f"Cannot parse '{date_col}' as dates.")
        return None
    y_cols = [c for c in df.columns if "donation" in c.lower() and "rwf" in c.lower()]
    if not y_cols:
        st.error("No donation column named 'total_donations_rwf' found.")
        return None
    y_col = y_cols[0]
    df = df.rename(columns={date_col: "ds", y_col: "y"})
    return df

#----------------------------------
# PDF Report Builder
#----------------------------------
def create_pdf_report(figs, titles):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for fig, title in zip(figs, titles):
        try:
            img = fig.to_image(format="png")
        except RuntimeError:
            # Fallback: save HTML snapshot as PNG is unavailable
            st.warning("PNG export requires 'kaleido' with Chrome installed. Skipping image in PDF.")
            img = None
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        if img:
            pdf.image(io.BytesIO(img), x=10, y=30, w=pdf.w-20)
        else:
            pdf.set_font("Arial", size=12)
            pdf.ln(20)
            pdf.cell(0, 10, "[Chart omitted: install 'kaleido' + Chrome to enable PNG export]", ln=True)
    return pdf.output(dest="S").encode("latin-1")

#----------------------------------
# Main App
#----------------------------------
def main():
    st.set_page_config(page_title="Donation Forecast & Analytics", layout="wide")
    st.title("📊 Donation Forecast & Analytics Dashboard")

    st.sidebar.header("Data & Filters")
    uploaded = st.sidebar.file_uploader("Upload donations CSV", type=["csv"])
    if not uploaded:
        st.info("Awaiting CSV upload…")
        return

    df = load_data(uploaded)
    if df is None:
        return

    # Filters
    donors = st.sidebar.multiselect("Donors", options=df.get("donor", pd.Series()).unique())
    campaigns = st.sidebar.multiselect("Campaign Types", options=df.get("campaign_type", pd.Series()).unique())
    regions = st.sidebar.multiselect("Regions", options=df.get("region", pd.Series()).unique())
    if donors or campaigns or regions:
        mask = pd.Series(True, index=df.index)
        if donors:
            mask &= df["donor"].isin(donors)
        if campaigns:
            mask &= df["campaign_type"].isin(campaigns)
        if regions:
            mask &= df["region"].isin(regions)
        df = df[mask]

    # Metrics
    total, count, avg = df["y"].sum(), df.shape[0], df["y"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Donations (RWF)", f"{total:,.0f}")
    c2.metric("Record Count", f"{count}")
    c3.metric("Avg Donation (RWF)", f"{avg:,.0f}")

    # Time Series
    st.subheader("Total Donations Over Time")
    fig1 = px.line(df, x="ds", y="y", labels={"ds":"Date","y":"Donations (RWF)"})
    st.plotly_chart(fig1, use_container_width=True)
    # Download logic with fallback
    try:
        buf1 = fig1.to_image(format="png")
        st.download_button("Download Time Series (PNG)", buf1, "time_series.png", "image/png")
    except RuntimeError:
        html1 = pio.to_html(fig1, full_html=False)
        st.download_button("Download Time Series (HTML)", html1, "time_series.html", "text/html")

    # Donor Breakdown
    if "donor" in df:
        st.subheader("Donations by Donor")
        donor_sum = df.groupby("donor")["y"].sum().reset_index()
        fig2 = px.bar(donor_sum, x="donor", y="y", labels={"donor":"Donor","y":"Donations (RWF)"})
        st.plotly_chart(fig2, use_container_width=True)
        try:
            buf2 = fig2.to_image(format="png")
            st.download_button("Download Donor Chart (PNG)", buf2, "donor_chart.png", "image/png")
        except RuntimeError:
            html2 = pio.to_html(fig2, full_html=False)
            st.download_button("Download Donor Chart (HTML)", html2, "donor_chart.html", "text/html")
    else:
        fig2 = None

    # Forecast
    st.sidebar.subheader("Forecast Horizon (months)")
    months = st.sidebar.slider("Select horizon", 1, 60, 12)
    if st.sidebar.button("Run Forecast"):
        with st.spinner("Fitting model…"):
            m = Prophet()
            m.fit(df[["ds","y"]])
            fut = m.make_future_dataframe(periods=months, freq='M')
            fc = m.predict(fut)
        st.subheader(f"{months}-Month Forecast")
        fig3 = px.line(fc, x="ds", y="yhat", labels={"ds":"Date","yhat":"Forecast (RWF)"})
        st.plotly_chart(fig3, use_container_width=True)
        try:
            buf3 = fig3.to_image(format="png")
            st.download_button("Download Forecast (PNG)", buf3, "forecast.png", "image/png")
        except RuntimeError:
            html3 = pio.to_html(fig3, full_html=False)
            st.download_button("Download Forecast (HTML)", html3, "forecast.html", "text/html")

        # PDF Report
        report = create_pdf_report([fig1, fig2 or fig1, fig3], ["Time Series", "Donor Breakdown", f"{months}-Month Forecast"])
        st.sidebar.download_button("Download Full Report (PDF)", report, "donation_report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
