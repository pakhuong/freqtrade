"""
Volume PairList provider

Provides dynamic pair list based on trade volumes
"""

import logging
from datetime import timedelta
from statistics import mean
from typing import Any, Literal, Optional

import pandas as pd
from cachetools import TTLCache
from scipy import stats

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_now, format_ms_time


logger = logging.getLogger(__name__)


class HqmPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "number_assets" not in self._pairlistconfig:
            raise OperationalException(
                "`number_assets` not specified. Please check your configuration "
                'for "pairlist.config.number_assets"'
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_pairs = self._pairlistconfig["number_assets"]
        self._refresh_period = self._pairlistconfig.get("refresh_period", 86400)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._lookback_timeframe = "1d"
        self._lookback_period = 35
        self._sort_direction: str | None = self._pairlistconfig.get("sort_direction", "desc")
        self._def_candletype = self._config["candle_type_def"]

        # get timeframe in minutes and seconds
        self._tf_in_min = timeframe_to_minutes(self._lookback_timeframe)
        _tf_in_sec = self._tf_in_min * 60

        # whether to use range lookback or not
        self._use_range = True

        if self._use_range & (self._refresh_period < _tf_in_sec):
            raise OperationalException(
                f"Refresh period of {self._refresh_period} seconds is smaller than one "
                f"timeframe of {self._lookback_timeframe}. Please adjust refresh_period "
                f"to at least {_tf_in_sec} and restart the bot."
            )

        if not self._use_range and not (
            self._exchange.exchange_has("fetchTickers")
            and self._exchange.get_option("ohlcv_has_history")
        ):
            raise OperationalException(
                "Exchange does not support dynamic whitelist in this configuration. "
                "Please edit your config and either remove TrendingPairList, "
                "or switch to using candles. and restart the bot."
            )

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return not self._use_range

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top high quality momentum pairs."

    @staticmethod
    def description() -> str:
        return "Provides dynamic pair list based on HQM (high quality momentum) pair."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "sort_direction": {
                "type": "option",
                "default": "desc",
                "options": ["", "asc", "desc"],
                "description": "Sort pairlist",
                "help": "Sort Pairlist ascending or descending by rate of change.",
            },
            **IPairList.refresh_period_parameter(),
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        # Generate dynamic whitelist
        # Must always run if this pairlist is not the first in the list.
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

            # No point in testing for blacklisted pairs...
            _pairlist = self.verify_blacklist(_pairlist, logger.info)

            if not self._use_range:
                filtered_tickers = [
                    v
                    for k, v in tickers.items()
                    if (
                        self._exchange.get_pair_quote_currency(k) == self._stake_currency
                        and (self._use_range or v.get(self._sort_key) is not None)
                        and v["symbol"] in _pairlist
                    )
                ]

                pairlist = [s["symbol"] for s in filtered_tickers]
            else:
                pairlist = _pairlist

            pairlist = self.filter_pairlist(pairlist, tickers)
            self._pair_cache["pairlist"] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        # Create bare minimum from tickers structure.
        filtered_tickers: list[dict[str, Any]] = [{"symbol": k} for k in pairlist]

        # get lookback period in ms, for exchange ohlcv fetch
        since_ms = (
            int(
                timeframe_to_prev_date(
                    self._lookback_timeframe,
                    dt_now()
                    + timedelta(
                        minutes=-(self._lookback_period * self._tf_in_min) - self._tf_in_min
                    ),
                ).timestamp()
            )
            * 1000
        )

        to_ms = (
            int(
                timeframe_to_prev_date(
                    self._lookback_timeframe, dt_now() - timedelta(minutes=self._tf_in_min)
                ).timestamp()
            )
            * 1000
        )

        # todo: utc date output for starting date
        self.log_once(
            f"Using trading range of {self._lookback_period} candles, timeframe: "
            f"{self._lookback_timeframe}, starting from {format_ms_time(since_ms)} "
            f"till {format_ms_time(to_ms)}",
            logger.info,
        )

        needed_pairs: ListPairsWithTimeframes = [
            (p, self._lookback_timeframe, self._def_candletype)
            for p in [s["symbol"] for s in filtered_tickers]
            if p not in self._pair_cache
        ]

        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms)

        hqm_columns = [
            "Symbol",
            "Price",
            "One-Day Price Return",
            "One-Day Return Percentile",
            "One-Week Price Return",
            "One-Week Return Percentile",
            "One-Month Price Return",
            "One-Month Return Percentile",
            "HQM Score",
        ]

        hqm_dataframe = pd.DataFrame(columns=hqm_columns)

        for i, p in enumerate(filtered_tickers):
            pair_candles = (
                candles[(p["symbol"], self._lookback_timeframe, self._def_candletype)]
                if (p["symbol"], self._lookback_timeframe, self._def_candletype) in candles
                else None
            )

            # in case of candle data calculate simple moving average and trend strength
            if (
                pair_candles is not None
                and not pair_candles.empty
                and not len(pair_candles.index) < self._lookback_period
            ):
                pair_candles["return_1d"] = pair_candles["close"].pct_change(1)
                pair_candles["return_1w"] = pair_candles["close"].pct_change(7)
                pair_candles["return_1m"] = pair_candles["close"].pct_change(30)

                new_hqm_row = pd.Series(
                    [
                        p["symbol"],
                        pair_candles["close"].iloc[-1],
                        pair_candles["return_1d"].iloc[-1],
                        "N/A",
                        pair_candles["return_1w"].iloc[-1],
                        "N/A",
                        pair_candles["return_1m"].iloc[-1],
                        "N/A",
                        "N/A",
                    ],
                    index=hqm_columns,
                )

                hqm_dataframe = pd.concat(
                    [hqm_dataframe, new_hqm_row.to_frame().T], ignore_index=True
                )

        time_periods = ["One-Day", "One-Week", "One-Month"]

        for row in hqm_dataframe.index:
            for time_period in time_periods:
                hqm_dataframe.loc[row, f"{time_period} Return Percentile"] = (
                    stats.percentileofscore(
                        hqm_dataframe[f"{time_period} Price Return"],
                        hqm_dataframe.loc[row, f"{time_period} Price Return"],
                    )
                )

        for row in hqm_dataframe.index:
            momentum_percentiles = []

            for time_period in time_periods:
                momentum_percentiles.append(
                    hqm_dataframe.loc[row, f"{time_period} Return Percentile"]
                )

            hqm_dataframe.loc[row, "HQM Score"] = round(mean(momentum_percentiles), 4)

        hqm_dataframe = hqm_dataframe.sort_values(by="HQM Score", ascending=False)
        sorted_tickers = hqm_dataframe["Symbol"].tolist()

        # Validate whitelist to only have active market pairs
        pairs = self._whitelist_for_active_markets([s for s in sorted_tickers])
        pairs = self.verify_blacklist(pairs, logmethod=logger.info)
        # Limit pairlist to the requested number of pairs
        pairs = pairs[: self._number_pairs]

        return pairs
