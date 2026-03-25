from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from aria.config import AriaConfig


@dataclass
class ReturnSeries:
    symbol: str
    returns: pd.Series


class FinancialDataClient:
    """
    Wrapper around yfinance and related utilities for benchmarking performance.
    """

    def __init__(self, config: AriaConfig) -> None:
        self._config = config

    def _enabled(self) -> bool:
        return bool(self._config.data_sources.financial_apis.yfinance)

    def price_history(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.
        """
        if not self._enabled():
            return pd.DataFrame()
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        return hist

    def total_return_series(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> ReturnSeries:
        """
        Compute simple returns from adjusted close prices.
        """
        hist = self.price_history(symbol, start=start, end=end, interval=interval)
        if hist.empty:
            return ReturnSeries(symbol=symbol, returns=pd.Series(dtype=float))
        returns = hist["Adj Close"].pct_change().dropna()
        return ReturnSeries(symbol=symbol, returns=returns)

    def annualized_return(
        self,
        series: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Compute annualized return from a series of periodic returns.
        """
        if series.empty:
            return 0.0
        cumulative = (1 + series).prod()
        n_periods = len(series)
        return cumulative ** (periods_per_year / n_periods) - 1

