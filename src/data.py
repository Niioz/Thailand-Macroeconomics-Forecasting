import pandas as pd
import numpy as np
import pandas_datareader.wb as wb
from datetime import datetime


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
        'FR.INR.LEND': 'Interest_Rate',
        'PA.NUS.FCRF': 'USD_THB'
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
        df_wb = df_wb.resample('MS').interpolate(method='linear', limit_area='inside')
    except Exception as e:
        raise DataUnavailableError(
            "Could not load World Bank macroeconomic history. Please refresh the data or try again later."
        ) from e

    if df_wb.dropna(subset=list(wb_series.values())).empty:
        raise DataUnavailableError(
            "World Bank returned no complete GDP, inflation, and interest-rate history. Please try again later."
        )
        
    df_macro = df_wb.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_macro) < 9:
        raise DataUnavailableError("World Bank returned fewer than nine complete monthly observations. Please try a longer history.")
    return df_macro


def load_set_history(csv_file, source="Investing.com"):
    """Read an Investing.com daily SET historical-data CSV without inventing prices."""
    try:
        prices = pd.read_csv(csv_file)
        prices.columns = prices.columns.str.strip().str.lstrip('\ufeff')
        if source == "SET":
            prices.rename(columns={'Close': 'Price', 'Closing': 'Price', 'Index': 'Price'}, inplace=True)
        if source == 'SET':
            prices.rename(columns={'Close': 'Price', 'Closing': 'Price', 'Index': 'Price'}, inplace=True)
        if not {'Date', 'Price'}.issubset(prices.columns):
            raise ValueError("The CSV must contain Date and Price columns from Investing.com.")
        dates = pd.to_datetime(prices['Date'], format='mixed', errors='coerce', dayfirst=(source == 'SET'))
        close = pd.to_numeric(prices['Price'].astype(str).str.replace(',', '', regex=False), errors='coerce')
        if dates.isna().any() or close.isna().any() or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError("The CSV contains invalid dates or closing prices. Please export daily SET history again.")
        history = pd.DataFrame({'Close': close.to_numpy()}, index=pd.DatetimeIndex(dates, name='Date')).sort_index()
        if history.index.has_duplicates:
            raise ValueError("The CSV contains duplicate dates. Please export daily history with one row per trading day.")
        if len(history) < 2:
            raise ValueError("Please upload at least two daily SET closing prices.")
        history['Return'] = history['Close'].pct_change(fill_method=None) * 100
        return history
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise ValueError("Could not read the Investing.com CSV. Please upload its original CSV export.") from error
