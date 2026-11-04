# Thailand Macro Forecasts and SET History

Streamlit dashboard for Thailand using World Bank macro data and optional Investing.com SET CSV uploads. Uses GDP (NY.GDP.MKTP.CD), inflation (FP.CPI.TOTL.ZG), lending interest rates (FR.INR.LEND), and official THB/USD exchange rates (PA.NUS.FCRF).

Annual observations are interpolated to monthly frequency within available annual anchors. Interpolated values are estimates, not observed monthly data. The latest complete history may lag the present. VAR forecasts GDP, inflation and lending rates; Prophet forecasts the official exchange rate.

Daily SET prices are not supplied by these World Bank indicators. Daily HMM/GARCH are excluded; uploaded SET history supports the monthly trading model described below. No synthetic daily market prices are generated.

Install with `pip install -r requirements.txt`, then run `streamlit run app.py`.

Click **Refresh data and retry** to clear cached downloads and forecasts.

Run regression checks with `python -m unittest discover -s tests -v`.

## SET price source

The dashboard accepts daily historical-data CSV exports from https://www.investing.com/indices/thailand-set-historical-data. Download the desired daily date range there and upload it using the SET uploader. Required columns are `Date` and `Price`; dates are sorted chronologically, comma-separated prices are normalized, and invalid prices or duplicate dates are rejected. Daily percentage returns are computed from uploaded closing prices. The first return is unavailable because there is no previous uploaded close.

Investing.com returned HTTP 403 during integration, so automatic scraping is not supported. Uploaded SET prices are displayed separately from World Bank macro forecasts. Random Forest, SHAP and the hold/cash backtest use uploaded SET history.

## Random Forest, SHAP and backtest

Upload daily SET CSV history using either official SET exports (Date and Price/Close/Closing/Index; day-first dates) or Investing.com exports (Date and Price). Official SET direct data access returned HTTP 403 during integration; no automatic feed is claimed. Obtain authorized historical data through SET's website when available.

The price-only Random Forest predicts a negative next-month return from trailing monthly returns, six-month volatility and distance from the twelve-month moving average. At least 48 usable labeled months and both target classes in training are required. SHAP contributions explain the latest fitted risk estimate. World Bank interpolations are excluded to avoid using later-published macro data in historical signals.

A chronological 60/40 split with a one-month gap reports annualized Sharpe (zero risk-free rate), total return and maximum drawdown against buy-and-hold. The strategy holds cash at risk >=50%, with zero cash yield, zero transaction costs and monthly rebalancing. The last uploaded month may be incomplete; displayed signal dates disclose data age.
