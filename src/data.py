import pandas as pd
import numpy as np
import pandas_datareader.wb as wb
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataUnavailableError(ValueError):
    """A data provider did not return usable history for the dashboard."""


def fetch_macro_data(years=10):
    end_date = datetime.today()
    start_year = end_date.year - years
    end_year = end_date.year
    
    # World Bank Data (Annual -> interpolate to monthly)
    wb_series = {
        'NY.GDP.MKTP.CD': 'GDP',
        'FP.CPI.TOTL.ZG': 'Inflation_YoY',  # Already YoY %
        'FR.INR.LEND': 'Interest_Rate'
    }
    
    try:
        # Fetch data
        df_wb = wb.download(indicator=list(wb_series.keys()), country=['TH'], start=start_year, end=end_year)
        df_wb.reset_index(inplace=True)
        
        # Clean and prepare for interpolation
        df_wb.drop(columns=['country'], inplace=True)
        df_wb['year'] = pd.to_datetime(df_wb['year'], format='%Y')
        df_wb.set_index('year', inplace=True)
        df_wb.rename(columns=wb_series, inplace=True)
        
        # World Bank data is returned descending (newest first). Sort it!
        df_wb.sort_index(inplace=True)
        
        # Resample to monthly and interpolate
        df_wb = df_wb.resample('MS').interpolate(method='linear')
    except Exception as e:
        raise DataUnavailableError(
            "Could not load World Bank macroeconomic history. Please refresh the data or try again later."
        ) from e

    if df_wb.dropna(subset=list(wb_series.values())).empty:
        raise DataUnavailableError(
            "World Bank returned no complete GDP, inflation, and interest-rate history. Please try again later."
        )
        
    # Yahoo Finance Data (Daily -> resample to monthly)
    yf_series = {
        'THB=X': 'USD_THB',
        '^SET.BK': 'SET_Index'
    }
    
    try:
        start_date = df_wb.index.min()
        df_yf = yf.download(
            list(yf_series.keys()), start=start_date, end=end_date,
            progress=False, auto_adjust=True
        )['Close']
        df_yf = df_yf.reindex(columns=list(yf_series)).replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        raise DataUnavailableError(
            "Could not load Yahoo Finance SET Index and USD/THB history. Please try again later."
        ) from e

    missing_history = [ticker for ticker in yf_series if df_yf[ticker].count() < 2]
    if missing_history:
        raise DataUnavailableError(
            "Yahoo Finance returned fewer than two closing prices for "
            f"{', '.join(missing_history)}. Historical prices are required; please refresh or try again later."
        )
    df_yf.rename(columns=yf_series, inplace=True)
    df_yf = df_yf.resample('MS').last() # End of month closing prices
        
    # Combine datasets
    df_macro = pd.concat([df_wb, df_yf], axis=1)
    
    # Drop rows beyond the last available World Bank macro data
    df_macro.dropna(subset=['GDP', 'Inflation_YoY', 'Interest_Rate'], inplace=True)
    
    # Feature Engineering
    # SET Index Returns and Downturn label
    df_macro['SET_Return_3M'] = df_macro['SET_Index'].pct_change(3, fill_method=None)
    df_macro['SET_Downturn'] = (df_macro['SET_Return_3M'] < 0).astype(int)
    
    # Drop rows with NaN returns (first 3 months)
    df_macro.dropna(inplace=True)

    if df_macro.empty:
        raise DataUnavailableError(
            "No overlapping macroeconomic and market history remains after preparing the data. "
            "Please refresh the data or try again later."
        )
    
    return df_macro

def fetch_daily_market_data(years=10):
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=years)
    try:
        df_daily = yf.download(
            '^SET.BK', start=start_date, end=end_date, progress=False,
            auto_adjust=True, multi_level_index=False
        )
    except Exception as e:
        raise DataUnavailableError(
            "Could not load daily SET Index history from Yahoo Finance. Please try again later."
        ) from e

    if df_daily is None or 'Close' not in df_daily.columns:
        raise DataUnavailableError("Yahoo Finance returned no SET Index closing prices. Please try again later.")
    df_daily = df_daily.replace([np.inf, -np.inf], np.nan).sort_index()
    if df_daily['Close'].count() < 2:
        raise DataUnavailableError(
            "Yahoo Finance returned fewer than two SET Index (^SET.BK) closing prices. "
            "Daily history is required; please refresh or try again later."
        )
    df_daily['Return'] = df_daily['Close'].pct_change(fill_method=None) * 100
    df_daily.dropna(subset=['Close', 'Return'], inplace=True)
    if df_daily.empty:
        raise DataUnavailableError("No usable daily SET Index returns are available. Please try again later.")
    return df_daily
