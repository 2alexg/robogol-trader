# strategies/super_scalper_strategy.py
#
# Author: Gemini
# Date: 2025-07-29

import pandas_ta as ta
from .base_strategy import BaseStrategy

class SuperScalperStrategy(BaseStrategy):
    """
    A scalping strategy based on the YouTube video by 'Trade Pro'.
    It uses a 200 EMA to determine the trend and Donchian Channels to
    identify entries on pullbacks to dynamic support/resistance.
    https://www.youtube.com/watch?v=TuNK4So36pw
    """
    is_multi_timeframe = False

    def __init__(self, params):
        super().__init__(params)
        self.ema_period = params['ema_period']
        self.donchian_period = params['donchian_period']
        self.ema_col = f"EMA_{self.ema_period}"
        self.dc_lower_col = f"DCL_{self.donchian_period}_{self.donchian_period}"
        self.dc_upper_col = f"DCU_{self.donchian_period}_{self.donchian_period}"

    def calculate_indicators(self, df_htf, df_ltf):
        df_ltf.ta.ema(length=self.ema_period, append=True)
        df_ltf.ta.donchian(lower_length=self.donchian_period, upper_length=self.donchian_period, append=True)
        return None, df_ltf

    def get_entry_signal(self, prev_row, current_row):
        is_uptrend = current_row['close'] > current_row[self.ema_col]
        touched_lower_band = current_row['low'] <= current_row[self.dc_lower_col]

        if is_uptrend and touched_lower_band:
            return 'LONG'

        is_downtrend = current_row['close'] < current_row[self.ema_col]
        touched_upper_band = current_row['high'] >= current_row[self.dc_upper_col]

        if is_downtrend and touched_upper_band:
            return 'SHORT'

        return None
