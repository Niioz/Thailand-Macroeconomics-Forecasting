from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.data import DataUnavailableError


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


class DashboardTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def tearDown(self):
        st.cache_data.clear()

    @patch('src.models.run_var_model')
    @patch('src.data.fetch_daily_market_data', return_value=pd.DataFrame())
    @patch('src.data.fetch_macro_data', return_value=pd.DataFrame())
    def test_empty_cached_data_stops_before_fitting_models(self, macro, daily, var):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertIn('No usable', app.error[0].value)
        var.assert_not_called()

    @patch('src.data.fetch_macro_data', side_effect=DataUnavailableError('Yahoo Finance history is unavailable.'))
    def test_provider_error_is_displayed_without_a_traceback(self, macro):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.error[0].value, 'Yahoo Finance history is unavailable.')

    @patch('src.data.fetch_daily_market_data', return_value=pd.DataFrame())
    @patch('src.data.fetch_macro_data')
    def test_retry_clears_cached_downloads(self, macro, daily):
        macro.side_effect = [pd.DataFrame(), DataUnavailableError('Fresh download attempted.')]
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(macro.call_count, 2)
        self.assertEqual(app.error[0].value, 'Fresh download attempted.')

    @patch('src.data.fetch_daily_market_data')
    @patch('src.data.fetch_macro_data')
    def test_valid_history_runs_the_complete_dashboard(self, macro, daily):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2016-01-01', periods=120, freq='MS')
        frame = pd.DataFrame({
            'GDP': 100 + rng.normal(size=120).cumsum(),
            'Inflation_YoY': 2 + rng.normal(scale=0.1, size=120).cumsum(),
            'Interest_Rate': 3 + rng.normal(scale=0.1, size=120).cumsum(),
            'USD_THB': 30 + rng.normal(scale=0.1, size=120).cumsum(),
            'SET_Index': 1000 * np.exp(rng.normal(scale=0.03, size=120).cumsum()),
        }, index=dates)
        frame['SET_Return_3M'] = frame['SET_Index'].pct_change(3)
        frame['SET_Downturn'] = (frame['SET_Return_3M'] < 0).astype(int)
        macro.return_value = frame.dropna()
        daily_returns = rng.normal(scale=1, size=300)
        daily.return_value = pd.DataFrame({
            'Close': 1000 * np.exp((daily_returns / 100).cumsum()),
            'Return': daily_returns,
        }, index=pd.date_range('2024-01-01', periods=300, freq='B'))
        app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(len(app.get('plotly_chart')), 6)


if __name__ == '__main__':
    unittest.main()
