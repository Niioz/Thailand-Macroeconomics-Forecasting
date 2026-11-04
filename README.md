# Thailand Macro Forecasts and SET History

Streamlit dashboard for Thailand using World Bank macro data and optional Investing.com SET CSV uploads. Uses GDP (NY.GDP.MKTP.CD), inflation (FP.CPI.TOTL.ZG), lending interest rates (FR.INR.LEND), and official THB/USD exchange rates (PA.NUS.FCRF).

Annual observations are interpolated to monthly frequency within available annual anchors. Interpolated values are estimates, not observed monthly data. The latest complete history may lag the present. VAR forecasts GDP, inflation and lending rates; Prophet forecasts the official exchange rate.

Daily SET prices are not supplied by these World Bank indicators. Daily HMM/GARCH, trading signals, SHAP and strategy backtests are excluded from the dashboard. No synthetic daily market prices are generated.

Install with `pip install -r requirements.txt`, then run `streamlit run app.py`.

Click **Refresh data and retry** to clear cached downloads and forecasts.

Run regression checks with `python -m unittest discover -s tests -v`.

## SET price source

The dashboard accepts daily historical-data CSV exports from https://www.investing.com/indices/thailand-set-historical-data. Download the desired daily date range there and upload it using the SET uploader. Required columns are `Date` and `Price`; dates are sorted chronologically, comma-separated prices are normalized, and invalid prices or duplicate dates are rejected. Daily percentage returns are computed from uploaded closing prices. The first return is unavailable because there is no previous uploaded close.

Investing.com returned HTTP 403 during integration, so automatic scraping is not supported. Uploaded SET prices are displayed separately from World Bank macro forecasts. Trading-model sections remain excluded.
