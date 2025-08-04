import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from prophet import Prophet

# Title
st.title("📊 Donation Forecast Dashboard - Maison Shalom")

# File uploader
uploaded_file = st.file_uploader("Upload Donation CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Basic checks and preprocessing
    if 'Date' not in df.columns or 'Amount' not in df.columns:
        st.error("CSV must contain 'Date' and 'Amount' columns.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

        # Optional filters
        with st.expander("🔍 Filter Data"):
            if 'Donor' in df.columns:
                selected_donors = st.multiselect("Select Donor(s):", df['Donor'].dropna().unique())
                if selected_donors:
                    df = df[df['Donor'].isin(selected_donors)]

            if 'Campaign Type' in df.columns:
                selected_campaigns = st.multiselect("Select Campaign Type(s):", df['Campaign Type'].dropna().unique())
                if selected_campaigns:
                    df = df[df['Campaign Type'].isin(selected_campaigns)]

            if 'Region' in df.columns:
                selected_regions = st.multiselect("Select Region(s):", df['Region'].dropna().unique())
                if selected_regions:
                    df = df[df['Region'].isin(selected_regions)]

        # Monthly Bar Chart
        df_monthly = df.groupby(df['Date'].dt.to_period("M")).agg({'Amount': 'sum'}).reset_index()
        df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()

        st.subheader("📈 Monthly Donations")
        fig1, ax1 = plt.subplots()
        sns.barplot(data=df_monthly, x='Date', y='Amount', ax=ax1)
        ax1.set_title("Monthly Donation Totals")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        st.pyplot(fig1)

        # Forecasting
        st.subheader("🔮 Forecast Future Donations")
        forecast_df = df[['Date', 'Amount']].rename(columns={'Date': 'ds', 'Amount': 'y'})

        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=6, freq='M')
        forecast = model.predict(future)

        fig2 = model.plot(forecast)
        st.pyplot(fig2)

        # Forecast chart download
        buf = io.BytesIO()
        fig2.savefig(buf, format="png")
        st.download_button("📥 Download Forecast Chart", data=buf.getvalue(), file_name="donation_forecast.png", mime="image/png")

        # Forecast CSV download
        forecast_csv = forecast.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast Data (CSV)", data=forecast_csv, file_name='donation_forecast.csv', mime='text/csv')

        # Forecast Excel download
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            forecast.to_excel(writer, index=False, sheet_name='Forecast')
        excel_buf.seek(0)
        st.download_button("📥 Download Forecast Data (Excel)", data=excel_buf, file_name="donation_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Raw data preview
        with st.expander("📄 Preview Cleaned Data"):
            st.dataframe(df.head(100))
