# strategies/macd_bollinger_atr_strategy.py
#
# Description:
# A mean-reversion strategy based on your custom logic:
# 1. Trend: MACD Histogram (hist > 0 is bullish, hist < 0 is bearish).
# 2. Entry: Bollinger Band mean-reversion.
#    - Bullish: Enter LONG on close below the lower band.
#    - Bearish: Enter SHORT on close above the upper band.
# 3. Exits: ATR-based Stop Loss and Take Profit.
#
# Author: Gemini
# Date: 2025-11-17

from .base_strategy import BaseStrategy
import pandas_ta as ta

class MacdBollingerAtrStrategy(BaseStrategy):
    """
    Implements the MACD-Trend, Bollinger-Entry, ATR-Exits strategy.
    """
    is_multi_timeframe = False

    def __init__(self, params):
        super().__init__(params)
        # Indicator params
        self.bb_length = int(params.get('bb_length', 20))
        self.bb_std = float(params.get('bb_std', 2.0))
        self.macd_fast = int(params.get('macd_fast', 12))
        self.macd_slow = int(params.get('macd_slow', 26))
        self.macd_signal = int(params.get('macd_signal', 9))
        self.atr_period = int(params.get('atr_period', 14))
        
        # Exit strategy params (now used as multipliers)
        self.atr_stop_loss_multiplier = float(params.get('atr_sl_multiplier', 2.0))
        self.rr_ratio = float(params.get('rr_ratio', 2.0))
        
        # Initialize column names to None
        self.macd_hist_col = None
        self.bbl_col = None
        self.bbu_col = None
        self.atr_col = None

    def calculate_indicators(self, high_tf_data, low_tf_data):
        df = low_tf_data.copy()
        
        # 1. MACD
        macd = df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)
        
        # 2. Bollinger Bands
        bb = df.ta.bbands(length=self.bb_length, std=self.bb_std, append=True)
        
        # 3. ATR
        atr = df.ta.atr(length=self.atr_period, append=True)
        
        # --- FIX: Dynamically find the column names ---
        # pandas-ta generates names like 'MACDh_12_26_9', 'BBL_20_2.0', 'ATRr_14'
        # But with optimization, floats can be long. We search for the prefix.
        
        # MACD Histogram
        for col in df.columns:
            if col.startswith('MACDh_'):
                self.macd_hist_col = col
                break
        
        # Bollinger Bands Lower and Upper
        for col in df.columns:
            if col.startswith('BBL_'):
                self.bbl_col = col
            elif col.startswith('BBU_'):
                self.bbu_col = col
                
        # ATR
        for col in df.columns:
            if col.startswith('ATRr_'):
                self.atr_col = col
                break

        # Sanity Check
        if not all([self.macd_hist_col, self.bbl_col, self.bbu_col, self.atr_col]):
             # If we can't find columns, something went wrong with calculation
             # This might happen if data is too short for indicators
             return None, None

        return None, df

    def get_entry_signal(self, prev_row, current_row):
        # Ensure columns were found before proceeding
        if not all([self.macd_hist_col, self.bbl_col, self.bbu_col]):
            return None

        # 1. Trend Condition (from MACD Histogram)
        is_bullish = current_row[self.macd_hist_col] > 0
        is_bearish = current_row[self.macd_hist_col] < 0
        
        # 2. Entry Condition (from Bollinger Bands)
        crossed_below_bb = current_row['close'] < current_row[self.bbl_col]
        crossed_above_bb = current_row['close'] > current_row[self.bbu_col]

        # Long Entry: Bullish trend + price closes below lower BB
        if is_bullish and crossed_below_bb:
            return 'LONG'
            
        # Short Entry: Bearish trend + price closes above upper BB
        if is_bearish and crossed_above_bb:
            return 'SHORT'
            
        return None

    # --- OVERRIDE: Custom ATR-based exit logic ---
    def calculate_exit_prices(self, entry_price, signal, current_row):
        """
        Calculates stop-loss and take-profit prices based on ATR.
        This overrides the default percentage-based method in BaseStrategy.
        """
        if not self.atr_col:
             return entry_price, entry_price # Safety fallback

        atr_value = current_row[self.atr_col]
        
        # Calculate stop loss distance based on ATR
        stop_loss_amount = atr_value * self.atr_stop_loss_multiplier
        
        # Calculate take profit distance based on SL (Risk/Reward Ratio)
        take_profit_amount = stop_loss_amount * self.rr_ratio

        if signal == 'LONG':
            stop_loss_price = entry_price - stop_loss_amount
            take_profit_price = entry_price + take_profit_amount
        else: # SHORT
            stop_loss_price = entry_price + stop_loss_amount
            take_profit_price = entry_price - take_profit_amount
            
        return stop_loss_price, take_profit_price
