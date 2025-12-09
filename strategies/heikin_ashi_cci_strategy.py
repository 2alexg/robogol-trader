# strategies/heikin_ashi_cci_strategy.py
#
# Description:
# A trend-following strategy using Heikin Ashi candles for entry signals and 
# dual CCI + EMA for trend filtering.
#
# Refined Logic (v2):
# 1. Candle Definitions:
#    - Bullish: No wick at bottom (Low == Open).
#    - Bearish: No wick at top (High == Open).
#    - Indecision: Wicks at both top and bottom.
# 2. Retracement Logic:
#    - Downside (Long Setup): Streak of Bearish & Indecision candles (no Bullish).
#      Must contain at least 3 Bearish candles.
#    - Upside (Short Setup): Streak of Bullish & Indecision candles (no Bearish).
#      Must contain at least 3 Bullish candles.
#
# Author: Gemini
# Date: 2025-12-06

import pandas as pd
import pandas_ta as ta
from .base_strategy import BaseStrategy

class HeikinAshiCciStrategy(BaseStrategy):
    """
    Heikin Ashi strategy with Dual CCI, EMA filter, and specific
    Retracement logic involving Indecision candles.
    """
    is_multi_timeframe = False

    def __init__(self, params):
        super().__init__(params)
        # Parameters
        self.ema_period = int(params.get('ema_period', 50))
        self.cci_fast_period = int(params.get('cci_fast_period', 14))
        self.cci_slow_period = int(params.get('cci_slow_period', 50))
        self.min_trend_candles = int(params.get('retracement_candles', 3))
        
        # Column placeholders
        self.ema_col = None
        self.cci_fast_col = None
        self.cci_slow_col = None

    def calculate_indicators(self, df_htf, df_ltf):
        df = df_ltf.copy()

        # 1. Calculate Heikin Ashi
        # Appends: HA_open, HA_high, HA_low, HA_close
        df.ta.ha(append=True)

        # 2. Calculate EMA (on standard Close)
        df.ta.ema(length=self.ema_period, append=True)
        self.ema_col = f"EMA_{self.ema_period}"

        # 3. Calculate Dual CCI
        cci_fast = df.ta.cci(length=self.cci_fast_period, append=True)
        cci_slow = df.ta.cci(length=self.cci_slow_period, append=True)
        
        # Dynamically identify columns
        for col in df.columns:
            if col.startswith(f"CCI_{self.cci_fast_period}_"):
                self.cci_fast_col = col
            elif col.startswith(f"CCI_{self.cci_slow_period}_"):
                self.cci_slow_col = col
        
        if not self.cci_fast_col or not self.cci_slow_col:
            raise ValueError("Could not calculate CCI columns.")

        # 4. Calculate CCI Trend (Fast - Slow)
        df['CCI_Trend'] = df[self.cci_fast_col] - df[self.cci_slow_col]

        # 5. Define Heikin Ashi Candle Types
        # Bullish: Low equals Open (No lower wick)
        df['HA_Bullish'] = df['HA_low'] == df['HA_open']
        
        # Bearish: High equals Open (No upper wick)
        df['HA_Bearish'] = df['HA_high'] == df['HA_open']
        
        # Indecision: Wicks at top and bottom (High > Open/Close AND Low < Open/Close)
        # Simplified: NOT Bullish AND NOT Bearish (Logic implies wicks on both sides if not flat)
        df['HA_Indecision'] = (~df['HA_Bullish']) & (~df['HA_Bearish'])

        # 6. Calculate Retracement Streaks (Vectorized)
        
        # --- Downside Retracement Calculation (Looking for LONG setup) ---
        # A valid downside retracement is a streak of "Bearish OR Indecision".
        # It is broken by a "Bullish" candle.
        # Within this streak, we need count(Bearish) >= 3.
        
        # Identify the "Breaker" of the streak (A Bullish Candle)
        df['Break_Downside_Streak'] = df['HA_Bullish']
        # Create a Group ID that increments every time the streak is broken
        downside_groups = df['Break_Downside_Streak'].cumsum()
        # Count cumulative Bearish candles within each group
        df['Downside_Bearish_Count'] = df.groupby(downside_groups)['HA_Bearish'].cumsum()
        # Mark rows where we are currently in a valid downside retracement
        # (Must NOT be the breaker candle itself, and must have enough bearish candles)
        df['Valid_Downside_Retracement'] = (~df['HA_Bullish']) & (df['Downside_Bearish_Count'] >= self.min_trend_candles)

        # --- Upside Retracement Calculation (Looking for SHORT setup) ---
        # A valid upside retracement is a streak of "Bullish OR Indecision".
        # It is broken by a "Bearish" candle.
        # Within this streak, we need count(Bullish) >= 3.
        
        # Identify the "Breaker" of the streak (A Bearish Candle)
        df['Break_Upside_Streak'] = df['HA_Bearish']
        # Create a Group ID
        upside_groups = df['Break_Upside_Streak'].cumsum()
        # Count cumulative Bullish candles within each group
        df['Upside_Bullish_Count'] = df.groupby(upside_groups)['HA_Bullish'].cumsum()
        # Mark rows where we are currently in a valid upside retracement
        df['Valid_Upside_Retracement'] = (~df['HA_Bearish']) & (df['Upside_Bullish_Count'] >= self.min_trend_candles)

        return None, df

    def get_entry_signal(self, prev_row, current_row):
        # --- Common Variables ---
        close = current_row['close']
        ema = current_row[self.ema_col]
        cci_fast = current_row[self.cci_fast_col]
        cci_trend = current_row['CCI_Trend']
        
        # --- LONG Logic ---
        # 1. Zone Filters: Close > EMA, CCI Fast > 0, CCI Trend > 0
        in_bullish_zone = (close > ema) and (cci_fast > 0) and (cci_trend > 0)
        
        if in_bullish_zone:
            # 2. Retracement: The PREVIOUS row must have been the end of a valid downside retracement
            valid_retracement_finished = prev_row['Valid_Downside_Retracement']
            
            # 3. Trigger: Current candle must be Bullish HA (The Reversal)
            trigger_candle = current_row['HA_Bullish']
            
            if valid_retracement_finished and trigger_candle:
                return 'LONG'

        # --- SHORT Logic ---
        # 1. Zone Filters: Close < EMA, CCI Fast < 0, CCI Trend < 0
        in_bearish_zone = (close < ema) and (cci_fast < 0) and (cci_trend < 0)
        
        if in_bearish_zone:
            # 2. Retracement: The PREVIOUS row must have been the end of a valid upside retracement
            valid_retracement_finished = prev_row['Valid_Upside_Retracement']
            
            # 3. Trigger: Current candle must be Bearish HA (The Reversal)
            trigger_candle = current_row['HA_Bearish']
            
            if valid_retracement_finished and trigger_candle:
                return 'SHORT'

        return None
