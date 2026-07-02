import pandas as pd
import numpy as np
import pandas_datareader.wb as wb
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

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
        print(f"Error fetching World Bank data: {e}")
        return None
        
    # Yahoo Finance Data (Daily -> resample to monthly)
    yf_series = {
        'THB=X': 'USD_THB',
        '^SET.BK': 'SET_Index'
    }
    
    try:
        start_date = df_wb.index.min()
        df_yf = yf.download(list(yf_series.keys()), start=start_date, end=end_date, progress=False)['Close']
        df_yf.rename(columns=yf_series, inplace=True)
        df_yf = df_yf.resample('MS').last() # End of month closing prices
    except Exception as e:
        print(f"Error fetching Yahoo Finance data: {e}")
        return None
        
    # Combine datasets
    df_macro = pd.concat([df_wb, df_yf], axis=1)
    
    # Drop rows beyond the last available World Bank macro data
    df_macro.dropna(subset=['GDP', 'Inflation_YoY', 'Interest_Rate'], inplace=True)
    
    # Feature Engineering
    # SET Index Returns and Downturn label
    df_macro['SET_Return_3M'] = df_macro['SET_Index'].pct_change(3)
    df_macro['SET_Downturn'] = (df_macro['SET_Return_3M'] < 0).astype(int)
    
    # Drop rows with NaN returns (first 3 months)
    df_macro.dropna(inplace=True)
    
    return df_macro

def fetch_daily_market_data(years=10):
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=years)
    try:
        df_daily = yf.download('^SET.BK', start=start_date, end=end_date, progress=False)
        df_daily['Return'] = df_daily['Close'].pct_change() * 100
        df_daily.dropna(inplace=True)
        return df_daily
    except Exception as e:
        print(f"Error fetching daily data: {e}")
        return None
