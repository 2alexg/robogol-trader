# strategies/base_strategy.py
#
# Description:
# This file contains the abstract BaseStrategy class that all other
# strategy classes must inherit from.
#
# Author: Gemini
# Date: 2025-11-17

class BaseStrategy:
    """
    A base class for all strategies, defining the required structure.
    """
    is_multi_timeframe = False

    def __init__(self, params):
        self.params = params

    def calculate_indicators(self, df_htf, df_ltf):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_entry_signal(self, prev_row, current_row):
        raise NotImplementedError("This method should be implemented by subclasses.")

    # --- NEW: Default exit logic for all existing strategies ---
    def calculate_exit_prices(self, entry_price, signal, current_row):
        """
        Calculates stop-loss and take-profit prices based on fixed percentages.
        This is the default method for simple strategies.
        More complex strategies (like ATR-based ones) will override this.
        """
        stop_loss_pct = self.params['stop_loss_percent']
        rr_ratio = self.params.get('rr_ratio', 2.0)
        take_profit_pct = stop_loss_pct * rr_ratio

        if signal == 'LONG':
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)
        else: # SHORT
            stop_loss_price = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 - take_profit_pct)
            
        return stop_loss_price, take_profit_price
