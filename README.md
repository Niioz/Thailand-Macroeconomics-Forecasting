# Thailand Macroeconomic Forecasts

World Bank-only Streamlit dashboard for Thailand. Uses GDP (NY.GDP.MKTP.CD), inflation (FP.CPI.TOTL.ZG), lending interest rates (FR.INR.LEND), and official THB/USD exchange rates (PA.NUS.FCRF).

Annual observations are interpolated to monthly frequency within available annual anchors. Interpolated values are estimates, not observed monthly data. The latest complete history may lag the present. VAR forecasts GDP, inflation and lending rates; Prophet forecasts the official exchange rate.

Daily SET prices are not supplied by these World Bank indicators. Daily HMM/GARCH, trading signals, SHAP and strategy backtests are excluded from the dashboard. No synthetic daily market prices are generated.

Install with `pip install -r requirements.txt`, then run `streamlit run app.py`.

Click **Refresh data and retry** to clear cached downloads and forecasts.

Run regression checks with `python -m unittest discover -s tests -v`.
