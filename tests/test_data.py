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
