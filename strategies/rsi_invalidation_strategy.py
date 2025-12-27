# strategies/rsi_invalidation_strategy.py
#
# Description:
# A trend-following RSI strategy with specific invalidation zones and ATR-based exits.
# - Entries are based on RSI retracements (crossing back from 40/60).
# - Moves are invalidated if RSI goes too deep (below 25 or above 75).
# - Invalidation resets only when RSI crosses the 50 line.
# - Exits are calculated dynamically using ATR (Average True Range).
#
# Author: Gemini
# Date: 2025-12-03

from .base_strategy import BaseStrategy
import pandas_ta as ta
import numpy as np

class RsiInvalidationStrategy(BaseStrategy):
    """
    RSI Strategy with "Invalidation" zones.
    1. Trend defined by SMA crossover (default 10 & 30).
    2. Enters on RSI crossing back from retracement levels (default 40 & 60).
    3. Invalidates signal if RSI hits extreme levels (default 25 & 75) 
       and waits for a reset to 50 before taking new trades.
    4. Exits calculated via ATR multiplier.
    """
    is_multi_timeframe = False

    def __init__(self, params):
        super().__init__(params)
        self.sma_fast_len = int(params.get('sma_fast', 10))
        self.sma_slow_len = int(params.get('sma_slow', 30))
        self.rsi_period = int(params.get('rsi_length', 14))

        # Optimization Parameters (Logic)
        # retracement_value: e.g., 40. Long triggers at 40, Short at 100-40=60.
        self.retracement_val = float(params.get('retracement_value', 40))
        # invalidation_value: e.g., 25. Long invalidates at 25, Short at 100-25=75.
        self.invalidation_val = float(params.get('invalidation_value', 25))

        # Optimization Parameters (Exits)
        self.atr_period = int(params.get('atr_period', 14))
        self.atr_sl_multiplier = float(params.get('atr_sl_multiplier', 2.0))
        self.rr_ratio = float(params.get('rr_ratio', 2.0))

        # Calculated Thresholds
        self.ret_low = self.retracement_val
        self.ret_high = 100 - self.retracement_val
        self.inv_low = self.invalidation_val
        self.inv_high = 100 - self.invalidation_val

        # Column Names
        self.sma_fast_col = f"SMA_{self.sma_fast_len}"
        self.sma_slow_col = f"SMA_{self.sma_slow_len}"
        self.rsi_col = f"RSI_{self.rsi_period}"
        self.taint_long_col = "taint_long"
        self.taint_short_col = "taint_short"
        self.atr_col = None # Determined dynamically

    def calculate_indicators(self, df_htf, df_ltf):
        df = df_ltf.copy()

        # 1. Calculate Standard Indicators
        df.ta.sma(length=self.sma_fast_len, append=True)
        df.ta.sma(length=self.sma_slow_len, append=True)
        df.ta.rsi(length=self.rsi_period, append=True)
        df.ta.atr(length=self.atr_period, append=True)

        # 2. Dynamically find ATR column (robustness for pandas_ta naming)
        for col in df.columns:
            if col.startswith('ATRr_') or col.startswith('ATR_'):
                self.atr_col = col
                break

        # 3. Calculate "Tainted" State for Invalidation Logic
        # We use a vectorised approach to maintain state across rows.
        # State: 1 = Tainted (Invalidated), -1 = Clean (Reset)

        # --- Long Invalidation Logic ---
        # Set taint (1) if RSI < 25. Clear taint (-1) if RSI > 50.
        df['taint_l_sig'] = 0
        df.loc[df[self.rsi_col] < self.inv_low, 'taint_l_sig'] = 1
        df.loc[df[self.rsi_col] > 50, 'taint_l_sig'] = -1
        # Forward fill the last known state. FillNA(-1) assumes we start clean.
        df[self.taint_long_col] = df['taint_l_sig'].replace(0, np.nan).ffill().fillna(-1)

        # --- Short Invalidation Logic ---
        # Set taint (1) if RSI > 75. Clear taint (-1) if RSI < 50.
        df['taint_s_sig'] = 0
        df.loc[df[self.rsi_col] > self.inv_high, 'taint_s_sig'] = 1
        df.loc[df[self.rsi_col] < 50, 'taint_s_sig'] = -1
        # Forward fill
        df[self.taint_short_col] = df['taint_s_sig'].replace(0, np.nan).ffill().fillna(-1)

        # Cleanup temp columns
        df.drop(columns=['taint_l_sig', 'taint_s_sig'], inplace=True)

        return None, df

    def get_entry_signal(self, prev_row, current_row):
        # 1. Trend Filter
        is_uptrend = current_row[self.sma_fast_col] > current_row[self.sma_slow_col]
        is_downtrend = current_row[self.sma_fast_col] < current_row[self.sma_slow_col]

        # 2. RSI Cross Logic
        # Long: Crossed BACK UP over retracement low (e.g. 40)
        rsi_cross_up = (prev_row[self.rsi_col] < self.ret_low and
                        current_row[self.rsi_col] > self.ret_low)

        # Short: Crossed BACK DOWN under retracement high (e.g. 60)
        rsi_cross_down = (prev_row[self.rsi_col] > self.ret_high and
                          current_row[self.rsi_col] < self.ret_high)

        # 3. Invalidation Check
        # If taint column is 1, the move is invalidated. We need -1 (Clean).
        is_long_valid = current_row[self.taint_long_col] == -1
        is_short_valid = current_row[self.taint_short_col] == -1

        # 4. Final Signal
        if is_uptrend and rsi_cross_up and is_long_valid:
            return 'LONG'

        if is_downtrend and rsi_cross_down and is_short_valid:
            return 'SHORT'

        return None

    def calculate_exit_prices(self, entry_price, signal, current_row):
        """
        Calculates stop-loss and take-profit prices based on ATR.
        Overrides the default percentage-based method.
        """
        if not self.atr_col or self.atr_col not in current_row:
             return entry_price, entry_price # Safety fallback

        atr_value = current_row[self.atr_col]

        # Calculate stop loss distance based on ATR
        stop_loss_amount = atr_value * self.atr_sl_multiplier

        # Calculate take profit distance based on SL (Risk/Reward Ratio)
        take_profit_amount = stop_loss_amount * self.rr_ratio

        if signal == 'LONG':
            stop_loss_price = entry_price - stop_loss_amount
            take_profit_price = entry_price + take_profit_amount
        else: # SHORT
            stop_loss_price = entry_price + stop_loss_amount
            take_profit_price = entry_price - take_profit_amount

        return stop_loss_price, take_profit_price
