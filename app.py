import streamlit as st
import plotly.graph_objects as go
from src.data import fetch_macro_data
from src.models import run_var_model, run_prophet_model

st.set_page_config(page_title="Thailand Macro Forecasts", layout="wide")
st.title("Thailand Macroeconomic Forecasts")
st.caption("Source: World Bank. Annual observations are interpolated to monthly frequency; monthly values are estimates, not observed monthly data.")
st.info("Daily SET prices are unavailable from this source. Market regimes, trading signals, backtests and daily volatility forecasts are therefore unavailable.")

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
