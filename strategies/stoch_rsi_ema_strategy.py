# strategies/stoch_rsi_ema_strategy.py
#
# Description:
# A multi-timeframe strategy that uses a high-timeframe EMA as a trend
# filter and a low-timeframe Stochastic RSI for entry signals.
#
# Author: Gemini
# Date: 2025-11-14

from .base_strategy import BaseStrategy
import pandas_ta as ta
import pandas as pd

class StochRsiEmaStrategy(BaseStrategy):
    """
    A multi-timeframe strategy:
    - High Timeframe: EMA (e.g., 200) for trend direction.
    - Low Timeframe: Stochastic RSI for entry signals.
    """
    def __init__(self, params):
        super().__init__(params)
        self.stoch_k = int(params.get('stoch_k', 14))
        self.stoch_d = int(params.get('stoch_d', 3))
        self.stoch_smooth_k = int(params.get('stoch_smooth_k', 3))
        self.rsi_period = int(params.get('rsi_period', 14))
        self.ema_period = int(params.get('ema_period', 200))
        self.oversold_threshold = int(params.get('oversold', 20))
        self.overbought_threshold = int(params.get('overbought', 80))
        
        self.stoch_k_col = None
        self.stoch_d_col = None
        self.ema_col = None

    def calculate_indicators(self, high_tf_data, low_tf_data):
        if high_tf_data is None:
            # This strategy cannot function without high-timeframe data.
            return None, None 

        # 1. Calculate High-Timeframe EMA
        htf_df = high_tf_data.copy()
        htf_df.ta.ema(length=self.ema_period, append=True)
        self.ema_col = f"EMA_{self.ema_period}"
        htf_df = htf_df[[self.ema_col]]

        # 2. Calculate Low-Timeframe Stochastic RSI
        ltf_df = low_tf_data.copy()
        
        # This call modifies ltf_df in-place due to append=True
        # and also returns a DataFrame with the StochRSI columns.
        stoch_rsi = ltf_df.ta.stochrsi(
            k=self.stoch_k,
            d=self.stoch_d,
            smooth_k=self.stoch_smooth_k,
            rsi_length=self.rsi_period,
            append=True
        )
        
        # Dynamically find the column names
        # We can look in either `stoch_rsi` or `ltf_df` (since it was appended)
        for col in stoch_rsi.columns:
            if col.startswith('STOCHRSIk_'): self.stoch_k_col = col
            if col.startswith('STOCHRSId_'): self.stoch_d_col = col

        if not all([self.stoch_k_col, self.stoch_d_col]):
            raise ValueError("Could not find StochRSI columns.")
            
        # --- FIX: This line is redundant and caused the crash ---
        # ltf_df = ltf_df.join(stoch_rsi)
        # --- End of Fix ---
        
        # 3. Merge High-Timeframe data into Low-Timeframe
        # Convert index to pandas datetime if not already
        ltf_df.index = pd.to_datetime(ltf_df.index)
        htf_df.index = pd.to_datetime(htf_df.index)

        # Re-sample high-tf data to low-tf index, filling forward
        merged_df = pd.merge_asof(
            ltf_df,
            htf_df,
            left_index=True,
            right_index=True,
            direction='forward'
        )
        
        return high_tf_data, merged_df

    def get_entry_signal(self, prev_row, current_row):
        # StochRSI Crossover Logic
        stoch_k_cross_above_d = (
            prev_row[self.stoch_k_col] < prev_row[self.stoch_d_col] and
            current_row[self.stoch_k_col] > current_row[self.stoch_d_col]
        )
        stoch_k_cross_below_d = (
            prev_row[self.stoch_k_col] > prev_row[self.stoch_d_col] and
            current_row[self.stoch_k_col] < current_row[self.stoch_d_col]
        )
        
        # Trend Filter
        price_above_ema = current_row['close'] > current_row[self.ema_col]
        price_below_ema = current_row['close'] < current_row[self.ema_col]
        
        # Long Entry Signal
        if (price_above_ema and 
            stoch_k_cross_above_d and
            current_row[self.stoch_k_col] < self.oversold_threshold):
            return 'LONG'
            
        # Short Entry Signal
        if (price_below_ema and
            stoch_k_cross_below_d and
            current_row[self.stoch_k_col] > self.overbought_threshold):
            return 'SHORT'
            
        return None
