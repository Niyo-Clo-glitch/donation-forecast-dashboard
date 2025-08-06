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
    # Detect date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        st.error("No date-like column found. Provide a column with 'date'.")
        return None
    date_col = date_cols[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        st.error(f"Cannot parse dates in '{date_col}'.")
        return None
    # Detect donations column
    y_cols = [c for c in df.columns if "donation" in c.lower() and "rwf" in c.lower()]
    if not y_cols:
        st.error("No 'total_donations_rwf' column found.")
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
            img = None
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        if img:
            pdf.image(io.BytesIO(img), x=10, y=30, w=pdf.w-20)
        else:
            pdf.set_font("Arial", size=12)
            pdf.ln(20)
            pdf.cell(0, 10, "[Chart omitted: PNG export failed]", ln=True)
    return pdf.output(dest="S").encode("latin-1")

#----------------------------------
# Main App
#----------------------------------
def main():
    st.set_page_config(page_title="Donation Forecast & Analytics", layout="wide")
    st.title("📊 Donation Forecast & Analytics Dashboard")

    # Sidebar: Data upload
    st.sidebar.header("Data & Filters")
    uploaded = st.sidebar.file_uploader("Upload donations CSV", type=["csv"])
    if not uploaded:
        st.info("Please upload a CSV file to proceed.")
        return
    df = load_data(uploaded)
    if df is None:
        return

    # Sidebar: Filters inside expander
    filters_exp = st.sidebar.expander("Filters", expanded=True)
    donors = filters_exp.multiselect("Donors", options=df.get("donor", pd.Series()).dropna().unique())
    campaigns = filters_exp.multiselect("Campaign Types", options=df.get("campaign_type", pd.Series()).dropna().unique())
    regions = filters_exp.multiselect("Regions", options=df.get("region", pd.Series()).dropna().unique())

    # Apply filters
    if any([donors, campaigns, regions]):
        mask = pd.Series(True, index=df.index)
        if donors:
            mask &= df["donor"].isin(donors)
        if campaigns:
            mask &= df["campaign_type"].isin(campaigns)
        if regions:
            mask &= df["region"].isin(regions)
        df = df[mask]

    # Key metrics
    total, count, avg = df["y"].sum(), df.shape[0], df["y"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Donations (RWF)", f"{total:,.0f}")
    c2.metric("Record Count", f"{count}")
    c3.metric("Avg Donation (RWF)", f"{avg:,.0f}")

    # Plotly config for download button
    plot_config = {"modeBarButtonsToAdd": ["toImage"], "displaylogo": False}

    # Time series chart
    st.subheader("Total Donations Over Time")
    fig1 = px.line(df, x="ds", y="y", labels={"ds": "Date", "y": "Donations (RWF)"})
    st.plotly_chart(fig1, use_container_width=True, config=plot_config)

    # Donations by donor
    if "donor" in df.columns:
        st.subheader("Donations by Donor")
        donor_sum = df.groupby("donor")["y"].sum().reset_index()
        fig2 = px.bar(donor_sum, x="donor", y="y", labels={"donor": "Donor", "y": "Donations (RWF)"})
        st.plotly_chart(fig2, use_container_width=True, config=plot_config)
    else:
        fig2 = None

    # Forecast controls
    st.sidebar.subheader("Forecast Settings")
    months = st.sidebar.slider("Forecast horizon (months)", 1, 60, 12)
    if st.sidebar.button("Run Forecast"):
        with st.spinner("Training forecasting model…"):
            m = Prophet()
            m.fit(df[["ds", "y"]])
            fut = m.make_future_dataframe(periods=months, freq='M')
            fc = m.predict(fut)
        st.subheader(f"{months}-Month Forecast")
        fig3 = px.line(fc, x="ds", y="yhat", labels={"ds": "Date", "yhat": "Forecast (RWF)"})
        st.plotly_chart(fig3, use_container_width=True, config=plot_config)
    else:
        fig3 = None

    # PDF report download
    if any([fig1, fig2, fig3]):
        report = create_pdf_report([fig1, fig2 or fig1, fig3 or fig1],
                                   ["Time Series", "Donor Breakdown", f"{months}-Month Forecast"])
        st.sidebar.download_button(
            "Download Full PDF Report", report,
            file_name="donation_report.pdf", mime="application/pdf"
        )

if __name__ == "__main__":
    main()
