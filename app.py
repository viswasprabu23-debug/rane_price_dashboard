
import streamlit as st
import pandas as pd
import numpy as np

st.title("RANE - Raw Material Intelligence Dashboard")

st.write("Welcome to our interactive dashboard!")


import plotly.graph_objects as go
import plotly.express as px
import zipfile, io, warnings

from statsmodels.tsa.holtwinters import Holt

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Rane | Raw Material Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RANE | Raw Material Intelligence")
st.caption("Price Analytics • Anomaly Detection • Forecasting")

# ---------------- UPLOAD ZIP ----------------

uploaded = st.sidebar.file_uploader(
    "Upload your dataset ZIP",
    type=["zip"]
)

if uploaded is None:
    st.info("Upload your hackathon ZIP file to start exploring.")
    st.stop()

# ---------------- READ FILES ----------------

@st.cache_data
def load_files(zip_bytes):

    datasets = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:

        for filename in z.namelist():

            if filename.endswith("/"):
                continue

            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(z.open(filename))
                    if not df.empty:
                        datasets[filename] = df

                elif filename.lower().endswith((".xlsx", ".xls")):
                    sheets = pd.read_excel(
                        z.open(filename),
                        sheet_name=None
                    )

                    for sheet, df in sheets.items():
                        if not df.empty:
                            datasets[f"{filename} | {sheet}"] = df

            except Exception:
                pass

    return datasets

datasets = load_files(uploaded.getvalue())

if not datasets:
    st.error("No readable CSV or Excel datasets found.")
    st.stop()

# ---------------- COLUMN DETECTION ----------------

def prepare_data(df):

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    date_col = None

    for col in df.columns:
        if any(x in col.lower() for x in
               ["date", "month", "year", "period", "time"]):
            date_col = col
            break

    if date_col is None:
        for col in df.columns:
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().mean() > 0.7:
                date_col = col
                break

    if date_col is None:
        return None, None, []

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    for col in df.columns:

        if col == date_col:
            continue

        cleaned = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("%", "", regex=False)
        )

        converted = pd.to_numeric(cleaned, errors="coerce")

        if converted.notna().mean() > 0.5:
            df[col] = converted

    metrics = [
        c for c in df.select_dtypes(include="number").columns
        if df[c].nunique() > 1
    ]

    return df.sort_values(date_col), date_col, metrics


prepared = {}

for name, df in datasets.items():

    cleaned, date_col, metrics = prepare_data(df)

    if cleaned is not None and metrics:
        prepared[name] = {
            "df": cleaned,
            "date": date_col,
            "metrics": metrics
        }

if not prepared:
    st.error(
        "Could not identify date and numeric columns. "
        "Please check the source datasets."
    )
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("Dashboard Filters")

dataset = st.sidebar.selectbox(
    "Select material / dataset",
    list(prepared.keys())
)

info = prepared[dataset]

metric = st.sidebar.selectbox(
    "Select price / metric",
    info["metrics"]
)

months = st.sidebar.selectbox(
    "Forecast horizon",
    [3, 6, 12],
    index=1
)

scenario = st.sidebar.slider(
    "Price scenario change (%)",
    -30, 30, 0, 5
)

threshold = st.sidebar.slider(
    "Anomaly threshold",
    1.0, 4.0, 2.0, 0.5
)

# ---------------- PROCESS DATA ----------------

df = info["df"][[info["date"], metric]].dropna()

df = df.sort_values(info["date"])

series = (
    df.set_index(info["date"])[metric]
    .resample("MS").mean()
    .dropna()
)

if len(series) < 3:
    st.warning("Not enough observations for reliable forecasting.")
    st.stop()

# ---------------- KPI METRICS ----------------

latest = series.iloc[-1]
previous = series.iloc[-2]

change = (
    (latest - previous) / abs(previous) * 100
    if previous != 0 else 0
)

average = series.mean()

rolling_mean = series.rolling(6, min_periods=3).mean()
rolling_std = series.rolling(6, min_periods=3).std()

zscore = (
    (series - rolling_mean) /
    rolling_std.replace(0, np.nan)
)

anomalies = series[zscore.abs() > threshold]

st.subheader(f"Overview: {dataset}")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest Value", f"{latest:,.2f}")
c2.metric("Monthly Change", f"{change:+.2f}%")
c3.metric("Historical Average", f"{average:,.2f}")
c4.metric("Anomalies", len(anomalies))

# ---------------- PRICE TREND ----------------

st.subheader("1. Historical Price Movement")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    name="Historical Price",
    mode="lines+markers"
))

fig.add_trace(go.Scatter(
    x=anomalies.index,
    y=anomalies.values,
    name="Anomalies",
    mode="markers",
    marker=dict(color="red", size=11)
))

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title=metric
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FORECAST ----------------

st.subheader("2. Price Forecast")

try:
    model = Holt(
        series.astype(float),
        initialization_method="estimated"
    ).fit(optimized=True)

    forecast = model.forecast(months)

    scenario_forecast = forecast * (1 + scenario / 100)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name="Historical"
    ))

    fig2.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        name="Base Forecast",
        line=dict(dash="dash")
    ))

    fig2.add_trace(go.Scatter(
        x=scenario_forecast.index,
        y=scenario_forecast.values,
        name="Scenario Forecast"
    ))

    fig2.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=metric
    )

    st.plotly_chart(fig2, use_container_width=True)

    forecast_table = pd.DataFrame({
        "Date": forecast.index.strftime("%Y-%m"),
        "Base Forecast": forecast.values,
        "Scenario Forecast": scenario_forecast.values
    })

    st.dataframe(
        forecast_table.round(2),
        use_container_width=True
    )

except Exception as e:
    st.error(f"Forecasting failed: {e}")

# ---------------- MONTHLY CHANGES ----------------

st.subheader("3. Month-on-Month Price Changes")

monthly_change = series.pct_change() * 100

fig3 = px.bar(
    x=monthly_change.index,
    y=monthly_change.values,
    labels={"x": "Date", "y": "Change (%)"},
    title="Monthly Price Change (%)"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- PRICE DISTRIBUTION ----------------

st.subheader("4. Price Distribution")

fig4 = px.histogram(
    series,
    nbins=20,
    title="Historical Price Distribution"
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------- ANOMALY TABLE ----------------

st.subheader("5. Anomaly Detection")

if len(anomalies):

    anomaly_table = pd.DataFrame({
        "Date": anomalies.index,
        "Price": anomalies.values,
        "Z-Score": zscore.loc[anomalies.index].values
    })

    st.dataframe(
        anomaly_table.round(2),
        use_container_width=True
    )

else:
    st.success("No anomalies detected at the selected threshold.")

# ---------------- DOWNLOAD ----------------

st.subheader("6. Download Results")

csv = series.rename(metric).to_csv().encode("utf-8")

st.download_button(
    "Download Historical Data",
    data=csv,
    file_name="rane_price_history.csv",
    mime="text/csv"
)

st.caption(
    "Exploratory statistical forecasts. "
    "Validate frequency, units, model accuracy and assumptions "
    "before using results for business decisions."
)
