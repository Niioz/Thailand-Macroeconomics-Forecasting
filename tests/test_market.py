import unittest
from io import StringIO
import numpy as np
import pandas as pd
from src.data import load_set_history
from src.market import analyze_market

class MarketTests(unittest.TestCase):
    def test_official_set_csv(self):
        frame = load_set_history(StringIO('Date,Close\n04/11/2026,102\n03/11/2026,100\n'), source='SET')
        self.assertEqual(frame.index[0], pd.Timestamp('2026-11-03'))
        self.assertAlmostEqual(frame.iloc[-1]['Return'], 2)
    def test_risk_shap_and_backtest(self):
        rng = np.random.default_rng(42)
        history = pd.DataFrame({'Close': 1000*np.exp(rng.normal(0.0001,0.01,3000).cumsum())}, index=pd.date_range('2010-01-01',periods=3000,freq='B'))
        risk, date, shap, results, metrics = analyze_market(history)
        self.assertTrue(0 <= risk <= 1)
        self.assertEqual(len(shap),5)
        self.assertTrue(np.isfinite(metrics.to_numpy()).all())
        self.assertTrue((metrics.loc['Maximum drawdown'] <= 0).all())
        self.assertGreater(len(results),10)
        self.assertLessEqual(results.index[-1], history.index[-1]+pd.offsets.MonthBegin(1))
    def test_short_history_rejected(self):
        frame = pd.DataFrame({'Close':[100,101]},index=pd.date_range('2020-01-01',periods=2))
        with self.assertRaisesRegex(ValueError,'five years'): analyze_market(frame)
