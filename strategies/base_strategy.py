# strategies/base_strategy.py
#
# Description:
# This file contains the abstract BaseStrategy class that all other
# strategy classes must inherit from.
#
# Author: Gemini
# Date: 2025-07-29

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
