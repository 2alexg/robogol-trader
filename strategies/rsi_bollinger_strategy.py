# rsi_bollinger_strategy.py
#
# Description:
# A mean-reversion strategy using Bollinger Bands and RSI. The strategy is
# based on the YouTube video by Trade Pro:
# https://www.youtube.com/watch?v=SOS_YnPZSQo

# This version uses a robust dynamic discovery method for indicator column
# names to ensure compatibility with future library updates.
#
# Author: Gemini
# Date: 2025-10-04

from .base_strategy import BaseStrategy
import pandas_ta as ta

class RsiBollingerStrategy(BaseStrategy):
    """
    A mean-reversion strategy that enters trades when the price moves outside
    the Bollinger Bands and the RSI indicates an overbought/oversold condition.
    """
    def __init__(self, params):
        super().__init__(params)
        self.rsi_period = int(params.get('rsi_period', 14))
        self.rsi_oversold = int(params.get('rsi_oversold', 30))
        self.rsi_overbought = int(params.get('rsi_overbought', 70))
        self.bb_length = int(params.get('bb_length', 20))
        self.bb_std = float(params.get('bb_std', 2.0))
        
        # --- FIX: Initialize column names to None. They will be discovered dynamically. ---
        self.rsi_col = None
        self.bbl_col = None
        self.bbm_col = None
        self.bbu_col = None

    def calculate_indicators(self, high_tf_data, low_tf_data):
        df = low_tf_data.copy()
        
        # Calculate RSI
        df.ta.rsi(length=self.rsi_period, append=True)
        
        # Calculate Bollinger Bands
        df.ta.bbands(length=self.bb_length, std=self.bb_std, append=True)

        # --- FIX: Dynamically find the column names after they are created ---
        # This is robust and not dependent on the library's naming convention.
        for col in df.columns:
            if col.startswith('RSI_'): self.rsi_col = col
            if col.startswith('BBL_'): self.bbl_col = col
            if col.startswith('BBM_'): self.bbm_col = col
            if col.startswith('BBU_'): self.bbu_col = col

        # Add a sanity check to ensure all columns were found
        if not all([self.rsi_col, self.bbl_col, self.bbm_col, self.bbu_col]):
            raise ValueError("Could not dynamically find all required indicator columns in the DataFrame.")

        return high_tf_data, df

    def get_entry_signal(self, prev_row, current_row):
        """
        Determines the entry signal based on Bollinger Bands and RSI.
        """
        # Long Entry: Price crosses below the lower Bollinger Band and RSI is oversold.
        if (prev_row['close'] > prev_row[self.bbl_col] and
                current_row['close'] < current_row[self.bbl_col] and
                current_row[self.rsi_col] < self.rsi_oversold):
            return 'LONG'

        # Short Entry: Price crosses above the upper Bollinger Band and RSI is overbought.
        if (prev_row['close'] < prev_row[self.bbu_col] and
                current_row['close'] > current_row[self.bbu_col] and
                current_row[self.rsi_col] > self.rsi_overbought):
            return 'SHORT'

        return None

