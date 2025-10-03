# strategies/rsi_bollinger_strategy.py
#
# Description:
# A mean-reversion strategy based on the YouTube video by Trade Pro:
# https://www.youtube.com/watch?v=SOS_YnPZSQo
# It uses Bollinger Bands to identify price extremes and RSI to confirm
# overbought/oversold conditions.

import pandas_ta as ta
from .base_strategy import BaseStrategy

class RsiBollingerStrategy(BaseStrategy):
    """
    RSI + Bollinger Bands Mean Reversion Strategy.
    https://www.youtube.com/watch?v=SOS_YnPZSQo
    """
    is_multi_timeframe = False

    def __init__(self, params):
        super().__init__(params)
        self.bband_period = params.get('bband_period', 20)
        self.bband_std = params.get('bband_std', 2.0)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)

        # Define the column names for the indicators
        self.bbl_col = f'BBL_{self.bband_period}_{self.bband_std}' # Lower Band
        self.bbu_col = f'BBU_{self.bband_period}_{self.bband_std}' # Upper Band
        self.rsi_col = f'RSI_{self.rsi_period}'

    def calculate_indicators(self, df_htf, df_ltf):
        """
        Calculates Bollinger Bands and RSI.
        """
        # Calculate Bollinger Bands
        df_ltf.ta.bbands(
            length=self.bband_period,
            std=self.bband_std,
            append=True
        )
        # Calculate RSI
        df_ltf.ta.rsi(
            length=self.rsi_period,
            append=True
        )
        return None, df_ltf

    def get_entry_signal(self, prev_row, current_row):
        """
        Determines the entry signal based on the strategy rules.
        """
        # Long Entry Conditions:
        # 1. Previous close was below the lower Bollinger Band.
        # 2. Previous RSI was oversold.
        # 3. Current close has crossed back up inside the lower band.
        if (prev_row['close'] < prev_row[self.bbl_col] and
            prev_row[self.rsi_col] < self.rsi_oversold and
            current_row['close'] > current_row[self.bbl_col]):
            return 'LONG'

        # Short Entry Conditions:
        # 1. Previous close was above the upper Bollinger Band.
        # 2. Previous RSI was overbought.
        # 3. Current close has crossed back down inside the upper band.
        if (prev_row['close'] > prev_row[self.bbu_col] and
            prev_row[self.rsi_col] > self.rsi_overbought and
            current_row['close'] < current_row[self.bbu_col]):
            return 'SHORT'

        return None
