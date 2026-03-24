import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# -------------------
# TITLE
# -------------------
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

ts_data = load_data()

# -------------------
# SHOW DATA
# -------------------
st.subheader("📈 Historical Sales")
st.line_chart(ts_data)

# -------------------
# MODEL
# -------------------
st.subheader("🤖 Training Auto ARIMA Model...")
model = auto_arima(ts_data, seasonal=False, stepwise=True)

# -------------------
# FORECAST
# -------------------
n_periods = st.slider("Select forecast months", 1, 24, 12)

forecast = model.predict(n_periods=n_periods)

future_index = pd.date_range(start=ts_data.index[-1], periods=n_periods+1, freq='M')[1:]
forecast_series = pd.Series(forecast, index=future_index)

# -------------------
# PLOT
# -------------------
st.subheader("🔮 Forecast Result")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(ts_data, label='Historical')
ax.plot(forecast_series, label='Forecast')
ax.legend()

st.pyplot(fig)