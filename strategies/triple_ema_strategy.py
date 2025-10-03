# strategies/triple_ema_strategy.py
#
# Description:
# This file contains the Triple EMA "Crossover, Close, Continue" strategy,
# as described in the YouTube video.
#
# Author: Gemini
# Date: 2025-07-29

import pandas_ta as ta
from .base_strategy import BaseStrategy

class TripleEMAStrategy(BaseStrategy):
    """
    Implements the "Triple-C" scalping strategy.
    - A long-term EMA (e.g., 200) determines the trend.
    - A medium-term EMA (e.g., 21) acts as a dynamic support/resistance level.
    - A short-term EMA (e.g., 9) is used for the entry signal crossover.
    https://www.youtube.com/watch?v=gpNCa-KbOfg
    """
    is_multi_timeframe = False

    def __init__(self, params):
        super().__init__(params)
        self.long_ema_period = params['long_ema_period']
        self.medium_ema_period = params['medium_ema_period']
        self.short_ema_period = params['short_ema_period']
        # Define column names for clarity
        self.long_ema_col = f"EMA_{self.long_ema_period}"
        self.medium_ema_col = f"EMA_{self.medium_ema_period}"
        self.short_ema_col = f"EMA_{self.short_ema_period}"

    def calculate_indicators(self, df_htf, df_ltf):
        df_ltf.ta.ema(length=self.long_ema_period, append=True)
        df_ltf.ta.ema(length=self.medium_ema_period, append=True)
        df_ltf.ta.ema(length=self.short_ema_period, append=True)
        return None, df_ltf

    def get_entry_signal(self, prev_row, current_row):
        # --- Long Entry Conditions ---
        # 1. Trend is up (Price > Long EMA)
        is_uptrend = current_row['close'] > current_row[self.long_ema_col]
        # 2. Price pulled back to the medium EMA in the previous candle
        pullback_to_medium_ema = prev_row['low'] <= prev_row[self.medium_ema_col]
        # 3. Fast EMA crosses above Medium EMA for the entry signal
        ema_bullish_crossover = (prev_row[self.short_ema_col] < prev_row[self.medium_ema_col] and
                                 current_row[self.short_ema_col] > current_row[self.medium_ema_col])

        if is_uptrend and pullback_to_medium_ema and ema_bullish_crossover:
            return 'LONG'

        # --- Short Entry Conditions ---
        # 1. Trend is down (Price < Long EMA)
        is_downtrend = current_row['close'] < current_row[self.long_ema_col]
        # 2. Price pulled back to the medium EMA in the previous candle
        pullback_to_medium_ema_short = prev_row['high'] >= prev_row[self.medium_ema_col]
        # 3. Fast EMA crosses below Medium EMA for the entry signal
        ema_bearish_crossover = (prev_row[self.short_ema_col] > prev_row[self.medium_ema_col] and
                                 current_row[self.short_ema_col] < current_row[self.medium_ema_col])

        if is_downtrend and pullback_to_medium_ema_short and ema_bearish_crossover:
            return 'SHORT'

        return None
