# Thailand Quantitative Research Engine

An interactive macro quant dashboard for Thailand. Fetches real economic and market data, runs econometric and machine learning models, and displays results through a Streamlit web UI.

## Data Sources

| Source | Data | Frequency |
|---|---|---|
| World Bank (via `pandas-datareader`) | GDP, Inflation (YoY %), Lending Interest Rate | Annual (interpolated to monthly) |
| Yahoo Finance (via `yfinance`) | SET Index (^SET.BK), USD/THB FX (THB=X) | Daily (resampled to month-end for macro) |

## Models

| Model | Purpose |
|---|---|
| **Vector Autoregression (VAR)** | Forecasts GDP, Inflation, and Interest Rate jointly over 12 months using their historical relationships |
| **Prophet (Meta)** | Forecasts USD/THB exchange rate with yearly seasonality |
| **Random Forest Classifier** | Predicts probability of a 3-month SET Index downturn using lagged macro features |
| **SHAP (TreeExplainer)** | Explainability — shows which features drove the latest downturn probability |
| **Gaussian HMM (3-state)** | Detects market regime (Bear / Neutral / Bull) from daily SET returns |
| **GARCH(1,1)** | Forecasts 30-day forward annualized volatility with volatility clustering |

## Dashboard Modules

1. **Macro & Regime Status** — KPI cards showing current HMM regime, downturn risk, inflation, and interest rate
2. **Strategy Tear Sheet** — Backtest comparing a macro-timing strategy (cash when risk > 50%) vs buy-and-hold SET Index, with Sharpe ratio, total return, and max drawdown
3. **GARCH Volatility** — Historical rolling 21-day volatility overlaid with 30-day GARCH forecast
4. **AI Explainability (SHAP)** — Waterfall plot explaining the latest downturn probability prediction
5. **Macro Forecasts** — VAR forecasts for GDP, Inflation, Interest Rate and Prophet forecast for USD/THB

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

streamlit, pandas, numpy, yfinance, pandas-datareader, statsmodels, prophet, scikit-learn, plotly, hmmlearn, arch, shap

## Notes

- World Bank data is annual; monthly interpolation uses linear resampling. Forecasts using interpolated data should be interpreted with caution.
- The backtest assumes zero transaction costs and a single 60/40 train/test split — not a walk-forward analysis.
- This is a research prototype, not a live trading system.
