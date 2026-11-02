import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.data import fetch_macro_data, fetch_daily_market_data
from src.models import (run_var_model, run_prophet_model, predict_downturn_prob_and_shap, 
                        run_backtest, run_hmm_regime, run_garch)

st.set_page_config(page_title="Thailand Quant Research Engine", layout="wide")

st.title("🇹🇭 Thailand Quantitative Research Engine")
st.markdown("Advanced macro-regime detection, GARCH volatility forecasting, and machine learning trading signals.")

@st.cache_data(ttl=86400)
def load_all_data():
    df_macro = fetch_macro_data(years=10)
    df_daily = fetch_daily_market_data(years=10)
    return df_macro, df_daily

@st.cache_data(ttl=86400)
def run_all_models(df_macro, df_daily):
    var_forecast = run_var_model(df_macro)
    prophet_forecast = run_prophet_model(df_macro, col='USD_THB')
    prob, explainer, shap_vals, latest_data = predict_downturn_prob_and_shap(df_macro)
    bt_results, bt_metrics = run_backtest(df_macro)
    df_regime, current_regime = run_hmm_regime(df_daily)
    garch_forecast, hist_vol = run_garch(df_daily)
    
    return var_forecast, prophet_forecast, prob, explainer, shap_vals, latest_data, bt_results, bt_metrics, df_regime, current_regime, garch_forecast, hist_vol

if st.button("Refresh data and retry"):
    load_all_data.clear()
    run_all_models.clear()

with st.spinner("Fetching data and running quant models..."):
    try:
        df_macro, df_daily = load_all_data()
        if df_macro is None or df_daily is None or df_macro.empty or df_daily.empty:
            raise ValueError(
                "No usable macroeconomic or SET Index history is available. "
                "Please refresh the data or try again later."
            )
        outputs = run_all_models(df_macro, df_daily)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    (var_forecast, prophet_forecast, prob, explainer, shap_vals, latest_data, 
     bt_results, bt_metrics, df_regime, current_regime, garch_forecast, hist_vol) = outputs

# --- Module 1: Status & Regime ---
st.header("Module 1: Macro & Regime Status")
col1, col2, col3, col4 = st.columns(4)

latest_inf = df_macro['Inflation_YoY'].iloc[-1]
latest_rate = df_macro['Interest_Rate'].iloc[-1]

col1.metric("Current Market Regime (HMM)", current_regime)
col2.metric("Market Downturn Risk", f"{prob:.1%}", 
            delta="High Risk" if prob > 0.5 else "Low Risk", 
            delta_color="inverse")
col3.metric("Inflation (YoY)", f"{latest_inf:.2f}%")
col4.metric("Interbank Rate", f"{latest_rate:.2f}%")

st.divider()

# --- Module 2 & 3: Backtest and Volatility ---
c1, c2 = st.columns(2)

with c1:
    st.header("Module 2: Strategy Tear Sheet")
    st.markdown("Simulated performance: Random Forest signal (Hold vs Cash) vs SET Index.")
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_results['Date'], y=bt_results['Market_Cum'], name="Buy & Hold SET", line=dict(color='gray')))
    fig_bt.add_trace(go.Scatter(x=bt_results['Date'], y=bt_results['Strat_Cum'], name="AI Strategy", line=dict(color='green')))
    fig_bt.update_layout(title="Strategy vs Benchmark (Cumulative Return)", yaxis_title="Growth of $1", hovermode="x unified", margin=dict(t=40, b=0))
    st.plotly_chart(fig_bt, use_container_width=True)
    
    # Metrics Table
    m_data = pd.DataFrame({
        "Metric": ["Total Return", "Annualized Sharpe", "Max Drawdown"],
        "AI Strategy": [f"{bt_metrics['Strat_Total']:.1%}", f"{bt_metrics['Strat_Sharpe']:.2f}", f"{bt_metrics['Strat_MaxDD']:.1%}"],
        "Buy & Hold": [f"{bt_metrics['Market_Total']:.1%}", f"{bt_metrics['Market_Sharpe']:.2f}", f"{bt_metrics['Market_MaxDD']:.1%}"]
    })
    st.table(m_data.set_index("Metric"))

with c2:
    st.header("Module 3: GARCH Volatility")
    st.markdown("30-day forward-looking volatility forecast using GARCH(1,1).")
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df_daily.index[-252:], y=hist_vol[-252:], name="Historical Vol (21D)", line=dict(color='navy')))
    fig_vol.add_trace(go.Scatter(x=garch_forecast['Date'], y=garch_forecast['Forecast_Vol'], name="GARCH Forecast", line=dict(color='red', dash='dash')))
    fig_vol.update_layout(title="Annualized Market Volatility (%)", yaxis_title="Volatility (%)", hovermode="x unified", margin=dict(t=40, b=0))
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# --- Module 4: SHAP Explainability ---
st.header("Module 4: AI Explainability (SHAP)")
st.markdown("Why is the model predicting this exact downturn probability? (Positive values increase risk)")

if shap_vals is not None:
    # Use matplotlib for SHAP plot and pass to Streamlit
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_vals[0], 
                                         base_values=explainer.expected_value[1], 
                                         data=latest_data.iloc[0], 
                                         feature_names=latest_data.columns), show=False)
    st.pyplot(fig)
else:
    st.info("Not enough data points to generate SHAP explanations.")

st.divider()

# --- Module 5: Core Macro Forecasts ---
with st.expander("View Underlying Macro Forecasts (VAR & Prophet)", expanded=False):
    c3, c4 = st.columns(2)
    def plot_forecast(historical_idx, historical_val, forecast_idx, forecast_val, title, ylabel, name_hist, name_fore):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_idx, y=historical_val, name=name_hist, line=dict(color='navy')))
        fig.add_trace(go.Scatter(x=forecast_idx, y=forecast_val, name=name_fore, line=dict(color='red', dash='dash')))
        fig.update_layout(title=title, yaxis_title=ylabel, hovermode="x unified", margin=dict(t=40, b=0))
        return fig

    with c3:
        st.plotly_chart(plot_forecast(df_macro.index, df_macro['GDP'], var_forecast.index, var_forecast['GDP'], "GDP", "GDP", "Historical", "VAR Forecast"), use_container_width=True)
        st.plotly_chart(plot_forecast(df_macro.index, df_macro['Inflation_YoY'], var_forecast.index, var_forecast['Inflation_YoY'], "Inflation YoY (%)", "Inflation (%)", "Historical", "VAR Forecast"), use_container_width=True)
    with c4:
        st.plotly_chart(plot_forecast(df_macro.index, df_macro['Interest_Rate'], var_forecast.index, var_forecast['Interest_Rate'], "Interbank Rate (%)", "Rate (%)", "Historical", "VAR Forecast"), use_container_width=True)
        fig_thb = go.Figure()
        fig_thb.add_trace(go.Scatter(x=df_macro.index, y=df_macro['USD_THB'], name="Historical", line=dict(color='darkgreen')))
        fig_thb.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], name="Prophet Forecast", line=dict(color='orange', dash='dash')))
        fig_thb.update_layout(title="USD/THB Exchange Rate", yaxis_title="THB per USD", hovermode="x unified", margin=dict(t=40, b=0))
        st.plotly_chart(fig_thb, use_container_width=True)
