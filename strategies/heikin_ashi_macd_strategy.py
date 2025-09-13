# strategies/heikin_ashi_macd.py
#
# Author: Gemini
# Date: 2025-07-29

import pandas_ta as ta
from .base_strategy import BaseStrategy

class HeikinAshiMACDStrategy(BaseStrategy):
    """
    A strategy using Heikin-Ashi candles for smoothed trend visualization
    and a MACD crossover for entry signals.
    """
    is_multi_timeframe = False

    def __init__(self, params):
        super().__init__(params)
        self.macd_fast = params['macd_fast']
        self.macd_slow = params['macd_slow']
        self.macd_signal = params['macd_signal']
        self.macd_hist_col = f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"

    def calculate_indicators(self, df_htf, df_ltf):
        df_ltf.ta.ha(append=True)
        df_ltf.ta.macd(close=df_ltf['HA_close'], fast=self.macd_fast,
                        slow=self.macd_slow, signal=self.macd_signal, append=True)
        return None, df_ltf

    def get_entry_signal(self, prev_row, current_row):
        is_bullish_candle = current_row['HA_open'] == current_row['HA_low']
        macd_crossed_up = prev_row[self.macd_hist_col] < 0 and current_row[self.macd_hist_col] > 0

        is_bearish_candle = current_row['HA_open'] == current_row['HA_high']
        macd_crossed_down = prev_row[self.macd_hist_col] > 0 and current_row[self.macd_hist_col] < 0

        if is_bullish_candle and macd_crossed_up:
            return 'LONG'
        if is_bearish_candle and macd_crossed_down:
            return 'SHORT'
        return None
