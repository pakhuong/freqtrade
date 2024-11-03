"""
Volume PairList provider

Provides dynamic pair list based on trade volumes
"""

import logging
from datetime import timedelta
from typing import Any, Literal, Optional

from cachetools import TTLCache

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_now, format_ms_time


logger = logging.getLogger(__name__)


SORT_VALUES = ["trendStrength"]


class TrendingPairList(IPairList):
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
        self._sort_key: Literal["trendStrength"] = self._pairlistconfig.get(
            "sort_key", "trendStrength"
        )
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._ma_period = self._pairlistconfig.get("ma_period", 10)
        self._lookback_timeframe = self._pairlistconfig.get("lookback_timeframe", "1d")
        self._sort_direction: Optional[str] = self._pairlistconfig.get("sort_direction", "desc")
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

        if not self._validate_keys(self._sort_key):
            raise OperationalException(f"key {self._sort_key} not in {SORT_VALUES}")

        candle_limit = self._exchange.ohlcv_candle_limit(
            self._lookback_timeframe, self._config["candle_type_def"]
        )

        if self._ma_period <= 0:
            raise OperationalException("TrendFilter requires ma_period to be > 0")

        if self._ma_period > candle_limit:
            raise OperationalException(
                "TrendFilter requires ma_period to not "
                f"exceed exchange max request size ({candle_limit})"
            )

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return not self._use_range

    def _validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top {self._pairlistconfig['number_assets']} trending pairs."

    @staticmethod
    def description() -> str:
        return "Provides dynamic pair list based on trend strength."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "Number of assets",
                "help": "Number of assets to use from the pairlist",
            },
            "sort_key": {
                "type": "option",
                "default": "trendStrength",
                "options": SORT_VALUES,
                "description": "Sort key",
                "help": "Sort key to use for sorting the pairlist.",
            },
            **IPairList.refresh_period_parameter(),
            "lookback_timeframe": {
                "type": "string",
                "default": "",
                "description": "Lookback Timeframe",
                "help": "Timeframe to use for lookback.",
            },
            "ma_period": {
                "type": "number",
                "default": 10,
                "description": "MA Period",
                "help": "Moving average period to use for trend strength calculation.",
            },
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
        if self._use_range:
            # Create bare minimum from tickers structure.
            filtered_tickers: list[dict[str, Any]] = [{"symbol": k} for k in pairlist]

            # get lookback period in ms, for exchange ohlcv fetch
            since_ms = (
                int(
                    timeframe_to_prev_date(
                        self._lookback_timeframe,
                        dt_now()
                        + timedelta(
                            minutes=-((self._ma_period + 1) * self._tf_in_min) - self._tf_in_min
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
                f"Using trading range of {self._ma_period + 1} candles, timeframe: "
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

            for i, p in enumerate(filtered_tickers):
                pair_candles = (
                    candles[(p["symbol"], self._lookback_timeframe, self._def_candletype)]
                    if (p["symbol"], self._lookback_timeframe, self._def_candletype) in candles
                    else None
                )

                # in case of candle data calculate simple moving average and trend strength
                if pair_candles is not None and not pair_candles.empty:
                    pair_candles["ma"] = pair_candles["close"].rolling(self._ma_period).mean()

                    pair_candles["trendStrength"] = abs(
                        (pair_candles["ma"] - pair_candles["ma"].shift(1))
                        / pair_candles["ma"].shift(1)
                        * 100
                    )

                    filtered_tickers[i]["trendStrength"] = pair_candles["trendStrength"].iloc[-1]
                else:
                    filtered_tickers[i]["trendStrength"] = 0
        else:
            # Tickers mode - filter based on incoming pairlist.
            filtered_tickers = [v for k, v in tickers.items() if k in pairlist]

        sorted_tickers = sorted(
            filtered_tickers,
            reverse=self._sort_direction == "desc",
            key=lambda t: t[self._sort_key],
        )

        # Validate whitelist to only have active market pairs
        pairs = self._whitelist_for_active_markets([s["symbol"] for s in sorted_tickers])
        pairs = self.verify_blacklist(pairs, logmethod=logger.info)
        # Limit pairlist to the requested number of pairs
        pairs = pairs[: self._number_pairs]

        return pairs
