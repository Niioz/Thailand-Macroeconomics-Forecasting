"""Monthly SET-only risk model with an out-of-sample hold/cash backtest."""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


def market_features(history):
    close = history['Close'].resample('MS').last()
    returns = close.pct_change(fill_method=None)
    features = pd.DataFrame({
        'Return_1M': returns,
        'Return_3M': close.pct_change(3, fill_method=None),
        'Return_6M': close.pct_change(6, fill_method=None),
        'Volatility_6M': returns.rolling(6).std(),
        'Distance_to_MA12': close / close.rolling(12).mean() - 1,
    })
    forward = returns.shift(-1)
    return features.replace([np.inf, -np.inf], np.nan), forward


def analyze_market(history):
    features, forward = market_features(history)
    usable = features.dropna().index
    labeled = usable.intersection(forward.dropna().index)
    if len(labeled) < 48:
        raise ValueError('Upload at least five years of daily SET history (48 usable monthly training and test observations are required).')
    X = features.loc[labeled]
    y = (forward.loc[labeled] < 0).astype(int)
    split = int(len(X) * 0.6)
    train_X, train_y = X.iloc[:split-1], y.iloc[:split-1]
    if train_y.nunique() < 2 or y.nunique() < 2:
        raise ValueError('The training history needs both rising and falling months. Please upload a longer history.')
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(train_X, train_y)
    test_X = X.iloc[split:]
    probability = model.predict_proba(test_X)[:, list(model.classes_).index(1)]
    realized = forward.loc[test_X.index]
    strategy = realized * (probability < 0.5)
    results = pd.DataFrame({'SET buy and hold': (1+realized).cumprod(), 'Hold/cash strategy': (1+strategy).cumprod()})
    # Dates indicate when the following month's return is realized.
    results.index = results.index + pd.offsets.MonthBegin(1)
    def metrics(series):
        equity = (1+series).cumprod().to_numpy()
        peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        std = series.std()
        return [float(np.sqrt(12)*series.mean()/std) if std > 0 else 0.0, float(equity[-1]-1), float((equity/peak-1).min())]
    summary = pd.DataFrame({'SET buy and hold': metrics(realized), 'Hold/cash strategy': metrics(strategy)}, index=['Annualized Sharpe (zero risk-free rate)', 'Total return', 'Maximum drawdown'])
    model.fit(X, y)
    latest = features.loc[[usable[-1]]]
    class_index = list(model.classes_).index(1)
    risk = float(model.predict_proba(latest)[0, class_index])
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(latest)
    values = values[class_index][0] if isinstance(values, list) else values[0, :, class_index]
    contributions = pd.DataFrame({'Feature': latest.columns, 'SHAP contribution to downturn probability': values, 'Feature value': latest.iloc[0].to_numpy()}).set_index('Feature')
    return risk, latest.index[0], contributions, results, summary
