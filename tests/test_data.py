import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data import fetch_macro_data, DataUnavailableError

class DataTests(unittest.TestCase):
    @patch('src.data.wb.download')
    def test_world_bank_only_history(self, download):
        index = pd.MultiIndex.from_product([['Thailand'], ['2020','2021','2022']], names=['country','year'])
        download.return_value = pd.DataFrame({'NY.GDP.MKTP.CD':[1,2,3], 'FP.CPI.TOTL.ZG':[1,2,3], 'FR.INR.LEND':[3,4,5], 'PA.NUS.FCRF':[30,31,32]}, index=index)
        frame = fetch_macro_data()
        self.assertEqual(list(frame.columns), ['GDP','Inflation_YoY','Interest_Rate','USD_THB'])
        self.assertEqual(len(frame),25)
        self.assertEqual(frame.iloc[-1]['USD_THB'],32)
        self.assertIn('PA.NUS.FCRF',download.call_args.kwargs['indicator'])
    @patch('src.data.wb.download', side_effect=ConnectionError('offline'))
    def test_unavailable_source(self, download):
        with self.assertRaises(DataUnavailableError): fetch_macro_data()


class SetCsvTests(unittest.TestCase):
    def test_investing_export_is_sorted_and_returns_are_correct(self):
        from io import StringIO
        from src.data import load_set_history
        frame = load_set_history(StringIO('Date,Price\n"11/04/2026","1,020.00"\n"11/03/2026","1,000.00"\n'))
        self.assertEqual(frame.iloc[0]['Close'], 1000)
        self.assertAlmostEqual(frame.iloc[1]['Return'], 2)
        self.assertTrue(np.isnan(frame.iloc[0]['Return']))
    def test_invalid_and_duplicate_prices_are_rejected(self):
        from io import StringIO
        from src.data import load_set_history
        for content in ['Date,Price\n11/03/2026,0\n11/04/2026,100\n', 'Date,Price\n11/03/2026,100\n11/03/2026,101\n', 'Date,Price\ninvalid,100\n11/04/2026,101\n']:
            with self.subTest(content=content):
                with self.assertRaises(ValueError): load_set_history(StringIO(content))
