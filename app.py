import streamlit as st
import plotly.graph_objects as go
from src.data import fetch_macro_data, load_set_history
from src.models import run_var_model, run_prophet_model

st.set_page_config(page_title="Thailand Macro Forecasts", layout="wide")
st.title("Thailand Macroeconomic Forecasts")
st.caption("Source: World Bank. Annual observations are interpolated to monthly frequency; monthly values are estimates, not observed monthly data.")
st.subheader("SET Index history")
st.markdown("Source: [Investing.com SET historical data](https://www.investing.com/indices/thailand-set-historical-data). Export daily history from that page and upload the CSV below. Automatic downloads are currently blocked by the provider.")
set_csv = st.file_uploader("Upload Investing.com SET daily historical data", type=['csv'])
if set_csv is not None:
    try:
        set_history = load_set_history(set_csv)
        st.line_chart(set_history['Close'])
        st.caption(f"{len(set_history):,} daily closing prices, {set_history.index.min():%Y-%m-%d} to {set_history.index.max():%Y-%m-%d}. Uploaded prices are separate from the annual macro estimates.")
        with st.expander("View SET prices and daily returns (%)"):
            st.dataframe(set_history)
    except ValueError as error:
        st.error(str(error))

@st.cache_data(ttl=86400)
def load_data():
    return fetch_macro_data(years=20)

@st.cache_data(ttl=86400)
def forecasts(data):
    return run_var_model(data), run_prophet_model(data, col='USD_THB')

if st.button("Refresh data and retry"):
    load_data.clear()
    forecasts.clear()

try:
    with st.spinner("Loading World Bank data and forecasts..."):
        data = load_data()
        var, fx = forecasts(data)
except ValueError as error:
    st.error(str(error))
    st.stop()

st.caption(f"Latest complete annual anchor: {data.index[-1]:%Y-%m}. Forecasts start after the available history, which may lag the current date.")
columns = st.columns(4)
for column, field, label in zip(columns, ['GDP', 'Inflation_YoY', 'Interest_Rate', 'USD_THB'], ['GDP (current USD)', 'Inflation (%)', 'Lending interest rate (%)', 'Official exchange rate (THB per USD)']):
    column.metric(label, f"{data[field].iloc[-1]:,.2f}")
    prediction = fx.set_index('ds')['yhat'] if field == 'USD_THB' else var[field]
    fig = go.Figure()
    fig.add_scatter(x=data.index, y=data[field], name='World Bank (monthly interpolation)')
    fig.add_scatter(x=prediction.index, y=prediction, name='Forecast')
    fig.update_layout(title=label)
    st.plotly_chart(fig, width='stretch')
with st.expander("View source data"):
    st.dataframe(data)
