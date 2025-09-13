# strategies/ichimoku_adx.py
#
# Author: Gemini
# Date: 2025-07-29

import pandas_ta as ta
from .base_strategy import BaseStrategy

class IchimokuADXStrategy(BaseStrategy):
    """
    A multi-timeframe strategy using ADX on a higher timeframe to determine
    the trend, and the Ichimoku Cloud on a lower timeframe for entries.
    """
    is_multi_timeframe = True

    def __init__(self, params):
        super().__init__(params)
        self.adx_period = params['adx_period']
        self.adx_threshold = params['adx_threshold']

    def calculate_indicators(self, df_htf, df_ltf):
        df_htf.ta.adx(length=self.adx_period, append=True)
        df_ltf.ta.ichimoku(append=True)
        return df_htf, df_ltf

    def get_entry_signal(self, prev_row, current_row):
        is_uptrend = (current_row[f'ADX_{self.adx_period}'] > self.adx_threshold and
                      current_row[f'DMP_{self.adx_period}'] > current_row[f'DMN_{self.adx_period}'])
        is_downtrend = (current_row[f'ADX_{self.adx_period}'] > self.adx_threshold and
                        current_row[f'DMN_{self.adx_period}'] > current_row[f'DMP_{self.adx_period}'])

        price_just_crossed_above_cloud = (prev_row['close'] < prev_row['ISA_9_26_52'] and
                                          current_row['close'] > current_row['ISA_9_26_52'])
        price_just_crossed_below_cloud = (prev_row['close'] > prev_row['ISB_26_52'] and
                                          current_row['close'] < current_row['ISB_26_52'])

        if is_uptrend and price_just_crossed_above_cloud:
            return 'LONG'
        if is_downtrend and price_just_crossed_below_cloud:
            return 'SHORT'
        return None
