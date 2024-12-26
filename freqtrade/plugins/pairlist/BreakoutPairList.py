import logging
from datetime import timedelta
from typing import Any, Literal

from cachetools import TTLCache

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_now, format_ms_time


logger = logging.getLogger(__name__)

MODE_VALUES = ["breakout", "inside_day"]

class BreakoutPairList(IPairList):
    """
    Pair list filter to select pairs that show a breakout-like pattern:
    1. Current day's close is outside the previous day's high/low range.
    2. Current day's candle forms either:
       - Higher high and higher low compared to previous day, or
       - Lower high and lower low compared to previous day.
    """

    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._stake_currency = self._config["stake_currency"]
        self._mode: Literal["breakout", "inside_day"] = self._pairlistconfig.get("mode", "breakout")
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)

        # We need daily candles to check daily breakouts.
        self._lookback_timeframe = "1d"
        # We need at least 2 days of data (yesterday and today).
        self._candle_count = 2

        # Validate refresh period (1 day = 1440 min -> 86400 s)
        # Though it's not strictly required to be that long, let's just note it.
        self._tf_in_min = timeframe_to_minutes(self._lookback_timeframe)
        _tf_in_sec = self._tf_in_min * 60
        if self._refresh_period < _tf_in_sec:
            logger.info(
                f"Refresh period {self._refresh_period}s is less than one daily candle interval "
                f"({_tf_in_sec}s). This might cause incomplete data."
            )

        # Check if OHLCV history is available
        if not (self._exchange.get_option("ohlcv_has_history")):
            raise OperationalException(
                "Exchange does not support historical candles for dynamic pairlist selection."
            )

    @property
    def needstickers(self) -> bool:
        """
        We rely entirely on candle data, not ticker data.
        """
        return False

    def short_desc(self) -> str:
        """
        Short pairlist method description - used for startup-messages
        """
        return f"{self.name} - list of pairs with breakout/inside-day."

    @staticmethod
    def description() -> str:
        return "Provides dynamic pair list based on daily breakout or inside-day patterns."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "mode": {
                "type": "option",
                "default": "breakout",
                "options": MODE_VALUES,
                "description": "Mode",
                "help": "Mode to use for pair list generation.",
            },
            **IPairList.refresh_period_parameter(),
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be empty or unused here.
        :return: List of pairs
        """
        pairlist = self._pair_cache.get("pairlist")

        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()

        else:
            # Use fresh pairlist
            # Check if pair quote currency equals to the stake currency.
            _pairlist = [
                k
                for k in self._exchange.get_markets(
                    quote_currencies=[self._stake_currency], tradable_only=True, active_only=True
                ).keys()
            ]

            # Remove blacklisted pairs
            _pairlist = self.verify_blacklist(_pairlist, logger.info)

            pairlist = self.filter_pairlist(_pairlist, tickers)
            self._pair_cache["pairlist"] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        Filter and return pairs that meet the breakout criteria.
        """
        # Determine the period for candle retrieval
        # We need at least the last 2 daily candles (yesterday and today)
        since_ms = (
            int(
                timeframe_to_prev_date(
                    self._lookback_timeframe, dt_now() - timedelta(days=self._candle_count)
                ).timestamp()
            )
            * 1000
        )

        to_ms = int(timeframe_to_prev_date(self._lookback_timeframe, dt_now()).timestamp()) * 1000

        self.log_once(
            f"Fetching {self._candle_count} daily candles from {format_ms_time(since_ms)} "
            f"to {format_ms_time(to_ms)}",
            logger.info,
        )

        needed_pairs: ListPairsWithTimeframes = [
            (p, self._lookback_timeframe, self._config["candle_type_def"]) for p in pairlist
        ]

        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms)

        filtered_pairs = []

        for p in pairlist:
            pair_candles = candles.get(
                (p, self._lookback_timeframe, self._config["candle_type_def"])
            )

            if pair_candles is not None and len(pair_candles) >= self._candle_count:
                # We only need the last two rows (yesterday and current)
                last_two = pair_candles.iloc[-2:]
                prev_day = last_two.iloc[0]
                curr_day = last_two.iloc[1]

                prev_high = prev_day["high"]
                prev_low = prev_day["low"]
                curr_close = curr_day["close"]
                curr_high = curr_day["high"]
                curr_low = curr_day["low"]

                if self._mode == "breakout":
                    # Condition 1: Current day's close is outside the previous day's high/low range
                    outside_range = (curr_close > prev_high) or (curr_close < prev_low)

                    # Condition 2: Either (higher high & higher low) OR (lower high & lower low)
                    higher_high_low = (curr_high > prev_high) and (curr_low > prev_low)
                    lower_high_low = (curr_high < prev_high) and (curr_low < prev_low)

                    if outside_range and (higher_high_low or lower_high_low):
                        filtered_pairs.append(p)

                elif self._mode == "inside_day":
                    # Condition 1: Current day's close is inside the previous day's high/low range
                    inside_range = (curr_close < prev_high) and (curr_close > prev_low)

                    if inside_range:
                        filtered_pairs.append(p)

        # Validate whitelist to only have active market pairs
        filtered_pairs = self._whitelist_for_active_markets(filtered_pairs)
        filtered_pairs = self.verify_blacklist(filtered_pairs, logmethod=logger.info)

        return filtered_pairs
