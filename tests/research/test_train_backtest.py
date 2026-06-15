"""Unit tests for train.py, backtest.py, and rolling.py.

All qlib calls are mocked; no live qlib runtime required.
7 test cases covering the key contract points verified in the plan review.

Mocking strategy: because qlib imports are DEFERRED inside function bodies,
we patch via mock.patch.dict(sys.modules, ...) PLUS mock.patch on the
module's load_config where applicable. The module itself is always importable
without qlib.
"""

from __future__ import annotations

from unittest import mock

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pred(dates, instruments, scores):
    """Build a MultiIndex (datetime, instrument) pred Series."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), sym) for d, sym in zip(dates, instruments)],
        names=["datetime", "instrument"],
    )
    return pd.Series(scores, index=idx, dtype=float)


def _empty_pred():
    idx = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
    return pd.Series([], index=idx, dtype=float)


def _make_bench_multiindex_df():
    """MultiIndex DataFrame as returned by D.features for benchmark."""
    midx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2023-06-17"), "SH000300"),
            (pd.Timestamp("2023-07-03"), "SH000300"),
            (pd.Timestamp("2023-07-04"), "SH000300"),
        ],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"$close": [3800.0, 3810.0, 3820.0]}, index=midx)


def _base_cfg():
    return {
        "strategy": {"topk": 15},
        "cost_model": {
            "account": 300000,
            "benchmark": "SH000300",
            "slippage": 0.001,
            "trade_unit": 100,
            "open_cost": 0.00026,
            "close_cost": 0.00076,
            "min_cost": 5,
            "deal_price": "close",
            "limit_threshold": 0.099,
        },
    }


def _make_portfolio_df():
    idx = pd.DatetimeIndex([pd.Timestamp("2023-07-03"), pd.Timestamp("2023-07-04")])
    return pd.DataFrame({"return": [0.01, 0.005]}, index=idx)


def _make_window():
    return {"test_start": "2023-07-01", "test_end": "2023-12-31"}


# ---------------------------------------------------------------------------
# Tests for _apply_price_filter
# ---------------------------------------------------------------------------


class TestApplyPriceFilter:
    def test_empty_pred_short_circuit(self):
        """Empty pred is returned as-is without calling D.features."""
        import ashare_lab.research.train as train_mod

        pred = _empty_pred()
        mock_D = mock.MagicMock()
        qlib_data_mod = mock.MagicMock()
        qlib_data_mod.D = mock_D

        with mock.patch.dict(
            "sys.modules",
            {"qlib": mock.MagicMock(), "qlib.data": qlib_data_mod},
        ), mock.patch(
            "ashare_lab.research.train.load_config",
            return_value={"universe": {"exclude_close_above_cny": 300}},
        ):
            result = train_mod._apply_price_filter(pred, "2023-01-03", "2023-06-30", "csi500")

        mock_D.features.assert_not_called()
        assert result.empty
        assert list(result.index.names) == ["datetime", "instrument"]

    def test_price_threshold_filters_expensive_stocks(self):
        """Stocks with close > threshold are excluded; stocks <= threshold survive."""
        import ashare_lab.research.train as train_mod

        dates = ["2023-01-03"] * 3
        syms = ["CHEAP", "EXACT", "PRICEY"]
        scores = [1.0, 2.0, 3.0]
        pred = _make_pred(dates, syms, scores)

        close_midx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-01-03"), "CHEAP"),
                (pd.Timestamp("2023-01-03"), "EXACT"),
                (pd.Timestamp("2023-01-03"), "PRICEY"),
            ],
            names=["datetime", "instrument"],
        )
        close_df = pd.DataFrame({"$close": [100.0, 300.0, 400.0]}, index=close_midx)

        mock_D = mock.MagicMock()
        mock_D.features.return_value = close_df
        qlib_data_mod = mock.MagicMock()
        qlib_data_mod.D = mock_D

        with mock.patch.dict(
            "sys.modules",
            {"qlib": mock.MagicMock(), "qlib.data": qlib_data_mod},
        ), mock.patch(
            "ashare_lab.research.train.load_config",
            return_value={"universe": {"exclude_close_above_cny": 300}},
        ):
            result = train_mod._apply_price_filter(pred, "2023-01-03", "2023-01-03", "csi500")

        result_syms = result.index.get_level_values("instrument").tolist()
        assert "CHEAP" in result_syms
        assert "EXACT" in result_syms
        assert "PRICEY" not in result_syms


# ---------------------------------------------------------------------------
# Tests for run_backtest
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def _run_with_mocks(self, mock_portfolio_metric, mock_indicator):
        """Helper: call run_backtest with standard mocks and return results."""
        import ashare_lab.research.backtest as bt_mod

        bench_df = _make_bench_multiindex_df()
        pred = _make_pred(["2023-07-03"] * 5, list("ABCDE"), [1.0] * 5)

        mock_backtest_daily = mock.MagicMock(
            return_value=(mock_portfolio_metric, mock_indicator)
        )
        mock_D = mock.MagicMock()
        mock_D.features.return_value = bench_df

        mock_strategy_cls = mock.MagicMock()
        qlib_evaluate = mock.MagicMock()
        qlib_evaluate.backtest_daily = mock_backtest_daily
        qlib_strategy = mock.MagicMock()
        qlib_strategy.TopkDropoutStrategy = mock_strategy_cls
        qlib_data_mod = mock.MagicMock()
        qlib_data_mod.D = mock_D

        with mock.patch.dict(
            "sys.modules",
            {
                "qlib": mock.MagicMock(),
                "qlib.contrib.evaluate": qlib_evaluate,
                "qlib.contrib.strategy": qlib_strategy,
                "qlib.data": qlib_data_mod,
            },
        ), mock.patch(
            "ashare_lab.research.backtest.load_config", return_value=_base_cfg()
        ):
            result = bt_mod.run_backtest(
                _make_window(), pred, n_drop=1, slippage_override=None
            )
        return result

    def test_portfolio_metric_dict_unpacked(self):
        """When backtest_daily returns {"1day": df}, the dict is unpacked."""
        portfolio_df = _make_portfolio_df()
        result_portfolio, _, _ = self._run_with_mocks(
            mock_portfolio_metric={"1day": portfolio_df},
            mock_indicator={"1day": {}},
        )
        assert isinstance(result_portfolio, pd.DataFrame)
        assert "return" in result_portfolio.columns
        assert len(result_portfolio) == 2

    def test_bench_close_xs_unpack(self):
        """bench_close is xs-unpacked: single DatetimeIndex, no MultiIndex."""
        portfolio_df = _make_portfolio_df()
        _, bench_close, _ = self._run_with_mocks(
            mock_portfolio_metric={"1day": portfolio_df},
            mock_indicator={"1day": {}},
        )
        assert isinstance(bench_close, pd.Series)
        assert not isinstance(bench_close.index, pd.MultiIndex)

    def test_lot_skip_indicator_freq_dict_unpack(self):
        """indicator {"1day": {date: df}} -> lot_skip_count computed correctly."""
        portfolio_df = _make_portfolio_df()
        # 2 orders: one filled, one skipped (trade_amount=0 != order_amount=200).
        date_df = pd.DataFrame(
            {"order_amount": [100.0, 200.0], "trade_amount": [100.0, 0.0]}
        )
        mock_indicator = {"1day": {"2023-07-03": date_df}}
        _, _, lot_skip_count = self._run_with_mocks(
            mock_portfolio_metric={"1day": portfolio_df},
            mock_indicator=mock_indicator,
        )
        assert lot_skip_count == 1  # one row where |200 - 0| > 1e-6


# ---------------------------------------------------------------------------
# Tests for run_full_walk_forward
# ---------------------------------------------------------------------------


class TestRunFullWalkForward:
    def _make_window(self, wid: int) -> dict:
        return {
            "step": wid - 1,
            "window_id": wid,
            "train_start": "2018-01-01",
            "train_end": f"202{wid}-12-31",
            "valid_start": f"202{wid + 1}-01-01",
            "valid_end": f"202{wid + 1}-06-30",
            "test_start": f"202{wid + 1}-07-01",
            "test_end": f"202{wid + 1}-12-31",
            "is_complete": True,
        }

    def _pred_and_label(self):
        pred = _make_pred(["2022-07-03"] * 5, list("ABCDE"), [1.0] * 5)
        label = _make_pred(["2022-07-03"] * 5, list("ABCDE"), [0.1] * 5)
        return pred, label

    def _portfolio_and_bench(self):
        portfolio_df = pd.DataFrame(
            {"return": [0.01]},
            index=pd.DatetimeIndex([pd.Timestamp("2022-07-03")]),
        )
        bench_close = pd.Series(
            [100.0, 101.0],
            index=pd.DatetimeIndex(
                [pd.Timestamp("2022-06-19"), pd.Timestamp("2022-07-03")]
            ),
        )
        return portfolio_df, bench_close

    def test_window_skip_on_exception(self, tmp_path):
        """A window that raises is omitted from results; no None entries."""
        import ashare_lab.research.rolling as rolling_mod

        good_window = self._make_window(1)
        bad_window = self._make_window(2)
        pred, label = self._pred_and_label()
        portfolio_df, bench_close = self._portfolio_and_bench()

        def mock_train(window, exp_dir, universe):
            if window["window_id"] == 2:
                raise RuntimeError("simulated training failure")
            return (tmp_path / f"models/w{window['window_id']}.pkl", pred, label)

        def mock_backtest(window, pred, n_drop, slippage_override):
            return portfolio_df, bench_close, 0

        with mock.patch.object(
            rolling_mod, "get_all_windows", return_value=[good_window, bad_window]
        ), mock.patch.object(
            rolling_mod, "train_window", side_effect=mock_train
        ), mock.patch.object(
            rolling_mod, "run_backtest", side_effect=mock_backtest
        ), mock.patch(
            "ashare_lab.research.rolling.load_config",
            return_value={"walk_forward": {"min_windows": 5}},
        ):
            results = rolling_mod.run_full_walk_forward(
                exp_dir=tmp_path, n_drop=1, universe="csi500"
            )

        assert len(results) == 1
        assert results[0]["window_id"] == 1
        assert None not in results

    def test_post_loop_min_windows_warning(self, tmp_path, caplog):
        """When completed windows < min_windows, a warning is logged post-loop."""
        import logging

        import ashare_lab.research.rolling as rolling_mod

        windows = [self._make_window(i) for i in range(1, 3)]  # only 2 windows
        pred, label = self._pred_and_label()
        portfolio_df, bench_close = self._portfolio_and_bench()

        def mock_train(window, exp_dir, universe):
            return (tmp_path / f"w{window['window_id']}.pkl", pred, label)

        def mock_backtest(window, pred, n_drop, slippage_override):
            return portfolio_df, bench_close, 0

        with mock.patch.object(
            rolling_mod, "get_all_windows", return_value=windows
        ), mock.patch.object(
            rolling_mod, "train_window", side_effect=mock_train
        ), mock.patch.object(
            rolling_mod, "run_backtest", side_effect=mock_backtest
        ), mock.patch(
            "ashare_lab.research.rolling.load_config",
            return_value={"walk_forward": {"min_windows": 5}},
        ), caplog.at_level(
            logging.WARNING, logger="ashare_lab.research.rolling"
        ):
            results = rolling_mod.run_full_walk_forward(
                exp_dir=tmp_path, n_drop=1, universe="csi500"
            )

        assert len(results) == 2
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("only 2 windows completed" in str(m) for m in warning_msgs), (
            f"Expected 'only 2 windows completed' warning; got: {warning_msgs}"
        )
