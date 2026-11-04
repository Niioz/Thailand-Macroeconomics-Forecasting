import streamlit as st
import plotly.graph_objects as go
from src.data import fetch_macro_data, load_set_history
from src.models import run_var_model, run_prophet_model
from src.market import analyze_market

st.set_page_config(page_title="Thailand Macro Forecasts", layout="wide")
st.title("Thailand Macroeconomic Forecasts")
st.caption("Source: World Bank. Annual observations are interpolated to monthly frequency; monthly values are estimates, not observed monthly data.")
st.subheader("SET Index history")
st.markdown("Source: [Investing.com SET historical data](https://www.investing.com/indices/thailand-set-historical-data). Export daily history from that page and upload the CSV below. Automatic downloads are currently blocked by the provider.")
st.markdown("Official source: [SET historical market data](https://www.set.or.th/en/services/connectivity-and-data/data/historical). Direct history access was unavailable during integration; upload a CSV export with Date and Price or Close columns. SET dates use day/month/year.")
source = st.selectbox("CSV source", ['SET', 'Investing.com'])
set_csv = st.file_uploader("Upload daily SET historical data (CSV)", type=['csv'])
if set_csv is not None:
    try:
        set_history = load_set_history(set_csv, source=source)
        st.line_chart(set_history['Close'])
        st.caption(f"{len(set_history):,} daily closing prices, {set_history.index.min():%Y-%m-%d} to {set_history.index.max():%Y-%m-%d}. Uploaded prices are separate from the annual macro estimates.")
        st.subheader("Random Forest downturn risk and strategy backtest")
        st.caption("Predicts a negative next-month SET return using past monthly SET returns, volatility and moving-average distance. The latest signal is dated below and may lag today. This is a price-history model; World Bank macro estimates are not used for trading signals.")
        try:
            risk, signal_date, contributions, equity, metrics = analyze_market(set_history)
            st.metric("Next-month downturn probability", f"{risk:.1%}")
            st.caption(f"Signal month: {signal_date:%Y-%m}. The uploaded final month may be incomplete.")
            st.subheader("SHAP explanation")
            st.bar_chart(contributions[['SHAP contribution to downturn probability']])
            st.caption("Positive contributions increase downturn probability. SHAP explains the fitted model, not economic causation.")
            st.dataframe(contributions)
            st.subheader("Strategy backtest")
            st.line_chart(equity)
            st.dataframe(metrics)
            st.caption("Chronological 60/40 train/test split with a one-month gap. Hold SET when estimated risk is below 50%; otherwise hold cash. Monthly rebalancing; zero cash yield and transaction costs. Sharpe is annualized from monthly returns; drawdown includes initial capital. This research backtest is not live performance.")
        except ValueError as error:
            st.info(str(error))
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
