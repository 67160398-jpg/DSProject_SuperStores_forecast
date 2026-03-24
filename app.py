import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# -------------------
# TITLE
# -------------------
st.set_page_config(page_title="SuperStore Forecast", layout="wide")
st.title("📊 SuperStore Sales Forecast")

# -------------------
# LOAD DATA
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("SuperStoreOrders.csv")
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    df.set_index('order_date', inplace=True)

    df['sales'] = df['sales'].astype(str).str.replace(',', '').astype(float)

    ts_data = df['sales'].resample('M').sum()
    return ts_data

try:
    ts_data = load_data()
except:
    st.error("❌ Cannot load data file")
    st.stop()

# -------------------
# SHOW DATA
# -------------------
st.subheader("📈 Historical Sales")
st.line_chart(ts_data)

# -------------------
# TRAIN MODEL (CACHE)
# -------------------
@st.cache_resource
def train_model(ts_data):
    model = auto_arima(ts_data,
                       seasonal=True,
                       m=12,
                       stepwise=True,
                       trace=False)
    return model

with st.spinner("🤖 Training Auto ARIMA Model..."):
    model = train_model(ts_data)

# -------------------
# FORECAST CONTROL
# -------------------
st.subheader("⚙️ Forecast Settings")
n_periods = st.slider("Select forecast months", 1, 24, 12)

# -------------------
# FORECAST
# -------------------
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

future_index = pd.date_range(start=ts_data.index[-1],
                             periods=n_periods + 1,
                             freq='M')[1:]

forecast_series = pd.Series(forecast, index=future_index)

# -------------------
# PLOT
# -------------------
st.subheader("🔮 Forecast Result")

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(ts_data, label='Historical')
ax.plot(forecast_series, label='Forecast')

# Confidence Interval
ax.fill_between(future_index,
                conf_int[:, 0],
                conf_int[:, 1],
                alpha=0.2)

ax.set_title("Sales Forecast (Next Months)")
ax.legend()
ax.grid()

st.pyplot(fig)

# -------------------
# SHOW FORECAST DATA
# -------------------
st.subheader("📊 Forecast Data")
st.dataframe(forecast_series.to_frame(name="Forecast Sales"))

# -------------------
# MODEL INFO
# -------------------
st.subheader("🧠 Model Info")
st.write("ARIMA Order:", model.order)
st.write("Seasonal Order:", model.seasonal_order)
