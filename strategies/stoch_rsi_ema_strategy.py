# strategies/stoch_rsi_ema_strategy.py
#
# Description:
# A multi-timeframe strategy using a long-term EMA on a higher timeframe (HTF)
# for trend direction and the Stochastic RSI on a lower timeframe (LTF) for
# finding oversold/overbought entry points.
#
# Author: Gemini
# Date: 2025-09-13 (v4 - Dynamic column name discovery)

import pandas as pd
import pandas_ta as ta
from .base_strategy import BaseStrategy

class StochRsiEmaStrategy(BaseStrategy):
    """
    Implements the Stochastic RSI + EMA multi-timeframe strategy.
    https://www.youtube.com/watch?v=UH9lp6_t86Y
    """
    is_multi_timeframe = True

    def __init__(self, params):
        """
        Initializes the strategy with its specific parameters.
        """
        super().__init__(params)
        self.ema_period = params['ema_period']
        self.stoch_rsi_k = params['stoch_rsi_k']
        self.stoch_rsi_d = params['stoch_rsi_d']
        self.rsi_length = params['rsi_length']
        self.oversold_threshold = params.get('oversold_threshold', 20)
        self.overbought_threshold = params.get('overbought_threshold', 80)

        # The name of the EMA column on the HTF dataframe
        self.ema_col = f"EMA_{self.ema_period}"

        # --- FIX: Initialize column names to None ---
        # We will no longer predict the names. We will find them at runtime.
        self.stoch_k_col = None
        self.stoch_d_col = None

    def calculate_indicators(self, df_htf, df_ltf):
        """
        Calculates the necessary indicators and aligns the two timeframes.
        """
        if df_htf is None or df_ltf is None:
            return None, None

        # 1. Calculate indicators on their respective timeframes
        df_htf[self.ema_col] = ta.ema(df_htf['close'], length=self.ema_period)
        
        df_ltf.ta.stochrsi(k=self.stoch_rsi_k, d=self.stoch_rsi_d, rsi_length=self.rsi_length, append=True)

        # --- FIX: Dynamically find the indicator column names ---
        # This is a robust way to get the column names created by pandas-ta,
        # regardless of its internal naming convention.
        try:
            self.stoch_k_col = next(col for col in df_ltf.columns if col.startswith('STOCHRSIk_'))
            self.stoch_d_col = next(col for col in df_ltf.columns if col.startswith('STOCHRSId_'))
        except StopIteration:
            # This will happen if pandas-ta fails to create the columns.
            # We raise an error to stop the backtest, as it cannot proceed.
            raise ValueError("Could not find Stochastic RSI columns after calculation.")

        # 2. Forward-fill the HTF data to align with the LTF
        ltf_freq = self.params['ltf'].replace('m', 'min')
        resampled_htf = df_htf.resample(ltf_freq).ffill()

        # 3. Merge the aligned HTF data into the LTF dataframe
        aligned_df = pd.merge(df_ltf, resampled_htf[[self.ema_col]], 
                              left_index=True, right_index=True, how='left')
        
        aligned_df[self.ema_col] = aligned_df[self.ema_col].ffill()
        
        return df_htf, aligned_df

    def get_entry_signal(self, prev_row, current_row):
        """
        Determines the entry signal based on the strategy rules.
        """
        # --- LONG ENTRY CONDITIONS ---
        price_above_ema = current_row['close'] > current_row[self.ema_col]
        stoch_bullish_cross = (prev_row[self.stoch_k_col] <= prev_row[self.stoch_d_col] and
                               current_row[self.stoch_k_col] > current_row[self.stoch_d_col])
        in_oversold_area = current_row[self.stoch_k_col] < self.oversold_threshold

        if price_above_ema and stoch_bullish_cross and in_oversold_area:
            return 'LONG'

        # --- SHORT ENTRY CONDITIONS ---
        price_below_ema = current_row['close'] < current_row[self.ema_col]
        stoch_bearish_cross = (prev_row[self.stoch_k_col] >= prev_row[self.stoch_d_col] and
                               current_row[self.stoch_k_col] < current_row[self.stoch_d_col])
        in_overbought_area = current_row[self.stoch_k_col] > self.overbought_threshold
        
        if price_below_ema and stoch_bearish_cross and in_overbought_area:
            return 'SHORT'
            
        return None

