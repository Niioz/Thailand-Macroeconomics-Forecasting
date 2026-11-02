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
