from pathlib import Path
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.data import DataUnavailableError

class DashboardTests(unittest.TestCase):
    @patch('src.data.fetch_macro_data', side_effect=DataUnavailableError('World Bank unavailable'))
    def test_provider_error_is_visible(self, download):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.error[0].value,'World Bank unavailable')
