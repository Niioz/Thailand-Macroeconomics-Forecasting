import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
import shap

def run_var_model(df, steps=12):
    var_cols = ['GDP', 'Inflation_YoY', 'Interest_Rate']
    df_var = df[var_cols].dropna()
    
    df_var_diff = df_var.diff().dropna()
    model = VAR(df_var_diff)
    try:
        results = model.fit(maxlags=12, ic='aic')
    except Exception:
        results = model.fit(maxlags=4)
        
    forecast_diff = results.forecast(df_var_diff.values[-results.k_ar:], steps=steps)
    last_obs = df_var.iloc[-1]
    forecast = pd.DataFrame(forecast_diff, columns=var_cols)
    forecast = forecast.cumsum() + last_obs.values
    
    future_dates = pd.date_range(start=df_var.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
    forecast.index = future_dates
    return forecast

def run_prophet_model(df, col='USD_THB', steps=12):
    df_p = df[[col]].reset_index()
    df_p.columns = ['ds', 'y']
    df_p = df_p.dropna()
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=steps, freq='MS')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)

def get_ml_features(df):
    df_ml = df.copy()
    features = ['GDP', 'Inflation_YoY', 'Interest_Rate', 'USD_THB']
    for col in features:
        df_ml[f'{col}_lag3'] = df_ml[col].shift(3)
        df_ml[f'{col}_lag6'] = df_ml[col].shift(6)
    df_ml.dropna(inplace=True)
    X = df_ml.drop(['SET_Downturn', 'SET_Index', 'SET_Return_3M'] + features, axis=1)
    y = df_ml['SET_Downturn']
    return X, y, df_ml

def predict_downturn_prob_and_shap(df):
    X, y, _ = get_ml_features(df)
    if len(X) < 10:
        return 0.5, None, None, None
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_scaled_df, y)
    
    latest_data = X_scaled_df.iloc[[-1]]
    prob = clf.predict_proba(latest_data)[0][1]
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(latest_data)
    
    # shap_values shape: (samples, features, classes) in newer SHAP
    if isinstance(shap_values, list):
        shap_vals_for_class = shap_values[1]
    else:
        shap_vals_for_class = shap_values[:, :, 1]
    
    return prob, explainer, shap_vals_for_class, latest_data

def run_backtest(df):
    """Backtest a strategy: Cash if probability of downturn > 0.5, else hold SET."""
    X, y, df_ml = get_ml_features(df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    # Train on first 60%, test on remaining 40% to simulate real performance
    split = int(len(X) * 0.6)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    
    # Align dates
    test_dates = df_ml.index[split:]
    # Calculate forward 1-month returns of SET Index
    df_ml['SET_Return_1M_Fwd'] = df_ml['SET_Index'].pct_change().shift(-1)
    returns = df_ml['SET_Return_1M_Fwd'].iloc[split:]
    
    # Strategy: 1 if prob < 0.5 (Hold), 0 if prob >= 0.5 (Cash)
    positions = (probs < 0.5).astype(int)
    strat_returns = positions * returns
    
    # Calculate cumulative returns
    cum_market = (1 + returns.fillna(0)).cumprod()
    cum_strat = (1 + strat_returns.fillna(0)).cumprod()
    
    # Calculate Sharpe Ratio (Annualized, assuming Monthly data)
    strat_sharpe = np.sqrt(12) * strat_returns.mean() / strat_returns.std() if strat_returns.std() > 0 else 0
    market_sharpe = np.sqrt(12) * returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Calculate Max Drawdown
    roll_max_strat = cum_strat.cummax()
    drawdown_strat = (cum_strat - roll_max_strat) / roll_max_strat
    max_dd_strat = drawdown_strat.min()
    
    roll_max_mkt = cum_market.cummax()
    drawdown_mkt = (cum_market - roll_max_mkt) / roll_max_mkt
    max_dd_mkt = drawdown_mkt.min()
    
    results = pd.DataFrame({
        'Date': test_dates,
        'Market_Cum': cum_market.values,
        'Strat_Cum': cum_strat.values
    })
    
    metrics = {
        'Strat_Sharpe': strat_sharpe, 'Market_Sharpe': market_sharpe,
        'Strat_MaxDD': max_dd_strat, 'Market_MaxDD': max_dd_mkt,
        'Strat_Total': cum_strat.iloc[-1] - 1, 'Market_Total': cum_market.iloc[-1] - 1
    }
    
    return results, metrics

def run_hmm_regime(df_daily):
    returns = df_daily['Return'].values.reshape(-1, 1)
    # Fit 3-state HMM
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(returns)
    hidden_states = model.predict(returns)
    
    # Order states by mean return (0: Bear, 1: Neutral, 2: Bull)
    means = model.means_.flatten()
    order = np.argsort(means)
    state_map = {order[0]: 'Bear/High Risk', order[1]: 'Neutral/Chop', order[2]: 'Bull/Growth'}
    
    current_state_idx = hidden_states[-1]
    current_regime = state_map[current_state_idx]
    
    df_daily['Regime'] = hidden_states
    return df_daily, current_regime

def run_garch(df_daily, steps=30):
    returns = df_daily['Return'].dropna()
    # GARCH(1,1) model
    am = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    res = am.fit(disp='off')
    forecasts = res.forecast(horizon=steps)
    # Annualized forecasted volatility
    vol_forecast = np.sqrt(forecasts.variance.values[-1, :]) * np.sqrt(252)
    
    future_dates = pd.date_range(start=df_daily.index[-1] + pd.DateOffset(days=1), periods=steps, freq='B')
    df_vol = pd.DataFrame({'Date': future_dates, 'Forecast_Vol': vol_forecast})
    
    # Historical rolling volatility
    hist_vol = returns.rolling(window=21).std() * np.sqrt(252)
    
    return df_vol, hist_vol
