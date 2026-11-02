import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from src.data import DataUnavailableError, fetch_daily_market_data, fetch_macro_data


def world_bank_history():
    index = pd.MultiIndex.from_product(
        [['Thailand'], ['2023', '2022', '2021', '2020', '2019', '2018']],
        names=['country', 'year'],
    )
    return pd.DataFrame({
        'NY.GDP.MKTP.CD': np.arange(6) * 1e10 + 4e11,
        'FP.CPI.TOTL.ZG': np.arange(6) * 0.2 + 1,
        'FR.INR.LEND': np.arange(6) * 0.1 + 3,
    }, index=index)


def yahoo_history():
    dates = pd.date_range('2018-01-01', '2023-01-01', freq='B')
    return pd.DataFrame({
        ('Close', 'THB=X'): np.linspace(30, 35, len(dates)),
        ('Close', '^SET.BK'): np.linspace(1200, 1600, len(dates)),
    }, index=dates)


class DataTests(unittest.TestCase):
    @patch('src.data.yf.download')
    @patch('src.data.wb.download')
    def test_single_set_quote_does_not_become_an_empty_macro_dataset(self, download_wb, download_yf):
        download_wb.return_value = world_bank_history()
        prices = yahoo_history()
        prices[('Close', '^SET.BK')] = np.nan
        prices.iloc[-1, prices.columns.get_loc(('Close', '^SET.BK'))] = 1500
        download_yf.return_value = prices
        with self.assertRaisesRegex(DataUnavailableError, r'fewer than two.*\^SET.BK'):
            fetch_macro_data()

    @patch('src.data.yf.download', return_value=pd.DataFrame())
    @patch('src.data.wb.download')
    def test_empty_yahoo_download_reports_a_provider_error(self, download_wb, download_yf):
        download_wb.return_value = world_bank_history()
        with self.assertRaisesRegex(DataUnavailableError, 'Yahoo Finance'):
            fetch_macro_data()

    @patch('src.data.wb.download', side_effect=ConnectionError('offline'))
    def test_world_bank_failure_reports_a_provider_error(self, download_wb):
        with self.assertRaisesRegex(DataUnavailableError, 'World Bank'):
            fetch_macro_data()

    @patch('src.data.yf.download')
    @patch('src.data.wb.download')
    def test_nonoverlapping_histories_are_rejected(self, download_wb, download_yf):
        download_wb.return_value = world_bank_history()
        prices = yahoo_history()
        prices.index = prices.index + pd.DateOffset(years=20)
        download_yf.return_value = prices
        with self.assertRaisesRegex(DataUnavailableError, 'No overlapping'):
            fetch_macro_data()

    @patch('src.data.yf.download')
    @patch('src.data.wb.download')
    def test_complete_history_retains_monthly_features(self, download_wb, download_yf):
        download_wb.return_value = world_bank_history()
        download_yf.return_value = yahoo_history()
        macro = fetch_macro_data()
        self.assertGreater(len(macro), 9)
        self.assertFalse(macro.isna().any().any())
        self.assertTrue(macro.index.is_monotonic_increasing)
        self.assertTrue((macro.index.day == 1).all())
        self.assertIn('SET_Return_3M', macro.columns)
        self.assertIn('SET_Downturn', macro.columns)

    @patch('src.data.yf.download')
    def test_empty_or_single_quote_daily_history_is_rejected(self, download_yf):
        for rows in (0, 1):
            with self.subTest(rows=rows):
                download_yf.return_value = pd.DataFrame(
                    {'Close': [1500.0] * rows},
                    index=pd.date_range('2023-01-01', periods=rows),
                )
                with self.assertRaisesRegex(DataUnavailableError, 'fewer than two SET Index'):
                    fetch_daily_market_data()

    @patch('src.data.yf.download')
    def test_daily_returns_do_not_depend_on_unrelated_missing_fields(self, download_yf):
        download_yf.return_value = pd.DataFrame(
            {'Close': [100.0, 102.0, 101.0], 'Volume': [np.nan] * 3},
            index=pd.date_range('2023-01-01', periods=3),
        )
        daily = fetch_daily_market_data()
        assert_allclose(daily['Return'].to_numpy(), [2.0, (101.0 / 102.0 - 1) * 100])
        self.assertFalse(isinstance(daily.columns, pd.MultiIndex))
        self.assertFalse(download_yf.call_args.kwargs['multi_level_index'])


if __name__ == '__main__':
    unittest.main()
