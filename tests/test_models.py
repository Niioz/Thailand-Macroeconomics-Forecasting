import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from src.models import run_var_model


def macro_history(rows=120):
    changes = np.random.default_rng(12).normal(size=(rows, 3))
    return pd.DataFrame(
        changes.cumsum(axis=0),
        columns=['GDP', 'Inflation_YoY', 'Interest_Rate'],
        index=pd.date_range('2010-01-01', periods=rows, freq='MS'),
    )


class VarModelTests(unittest.TestCase):
    def test_empty_and_short_histories_report_required_observations(self):
        for rows in (0, 1, 4, 8):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, f'at least 9.*found {rows}'):
                    run_var_model(macro_history(rows))

    def test_insufficient_history_after_cleaning_is_rejected(self):
        frame = macro_history(12)
        frame.iloc[:4, 0] = np.nan
        frame.iloc[4, 1] = np.inf
        with self.assertRaisesRegex(ValueError, 'at least 9.*found 7'):
            run_var_model(frame)

    def test_missing_macro_columns_are_identified(self):
        with self.assertRaisesRegex(ValueError, 'missing required columns: Interest_Rate'):
            run_var_model(macro_history().drop(columns='Interest_Rate'))

    def test_small_estimable_samples_produce_finite_monthly_forecasts(self):
        for rows in (9, 12, 20):
            with self.subTest(rows=rows):
                frame = macro_history(rows)
                original = frame.copy(deep=True)
                forecast = run_var_model(frame, steps=5)
                self.assertEqual(forecast.shape, (5, 3))
                self.assertTrue(np.isfinite(forecast.to_numpy()).all())
                self.assertEqual(forecast.index[0], frame.index[-1] + pd.offsets.MonthBegin())
                self.assertEqual(forecast.index.freqstr, 'MS')
                assert_frame_equal(frame, original)

    def test_zero_lag_selection_forecasts_mean_changes(self):
        # Independent monthly changes select VAR(0), which crashed in statsmodels 0.14.
        frame = macro_history()
        forecast = run_var_model(frame, steps=4)
        expected = frame.iloc[-1].to_numpy() + (
            np.arange(1, 5)[:, None] * frame.diff().dropna().mean().to_numpy()
        )
        assert_allclose(forecast.to_numpy(), expected)

    def test_singular_lag_selection_has_a_sample_bounded_fallback(self):
        with patch('src.models.VAR.select_order', side_effect=np.linalg.LinAlgError('singular')):
            forecast = run_var_model(macro_history(9))
        self.assertEqual(forecast.shape, (12, 3))
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())

    def test_unestimable_constant_changes_report_a_useful_error(self):
        frame = macro_history(20)
        frame.loc[:, :] = np.arange(20)[:, None]
        with self.assertRaisesRegex(ValueError, 'independent variation'):
            run_var_model(frame)


if __name__ == '__main__':
    unittest.main()
