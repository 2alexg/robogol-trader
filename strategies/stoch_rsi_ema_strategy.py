# strategies/stoch_rsi_ema_strategy.py
#
# Description:
# A multi-timeframe strategy based on the YouTube video by Matt Diamond:
# https://www.youtube.com/watch?v=UH9lp6_t86Y
# It uses a 200 EMA on a higher timeframe (e.g., 1h) to determine the
# overall trend, and a Stochastic RSI on a lower timeframe (e.g., 5m)
# to find pullback entry opportunities.

import pandas_ta as ta
from .base_strategy import BaseStrategy

class StochRsiEmaStrategy(BaseStrategy):
    """
    Stochastic RSI + EMA Multi-Timeframe Strategy.
    """
    is_multi_timeframe = True

    def __init__(self, params):
        super().__init__(params)
        self.htf = params.get('htf')
        self.ltf = params.get('ltf')
        self.ema_period = params.get('ema_period', 200)
        self.stoch_k = params.get('stoch_k', 14)
        self.stoch_d = params.get('stoch_d', 3)
        self.rsi_period = params.get('rsi_period', 14)
        self.stoch_oversold = params.get('stoch_oversold', 20)
        self.stoch_overbought = params.get('stoch_overbought', 80)

        # Define the column names for the indicators
        self.ema_col = f'EMA_{self.ema_period}'
        self.stoch_k_col = f'STOCHRSIk_{self.stoch_k}_{self.stoch_d}_{self.rsi_period}'
        self.stoch_d_col = f'STOCHRSId_{self.stoch_k}_{self.stoch_d}_{self.rsi_period}'

    def calculate_indicators(self, df_htf, df_ltf):
        """
        Calculates the EMA on the HTF and the StochRSI on the LTF.
        It then aligns the HTF data to the LTF for decision making.
        """
        if df_htf is None or df_ltf is None:
            return None, None

        # 1. Calculate EMA on the Higher Timeframe (HTF)
        df_htf.ta.ema(length=self.ema_period, append=True)
        htf_trend_data = df_htf[[self.ema_col, 'close']]
        htf_trend_data = htf_trend_data.rename(columns={'close': 'htf_close'})

        # 2. Calculate Stochastic RSI on the Lower Timeframe (LTF)
        df_ltf.ta.stochrsi(
            k=self.stoch_k,
            d=self.stoch_d,
            rsi_length=self.rsi_period,
            append=True
        )

        # 3. Align the dataframes
        # Convert the timeframe strings (e.g., '1h', '5m') to pandas frequency strings
        pandas_ltf_freq = self.ltf.replace('m', 'min').replace('h', 'H').replace('d', 'D')

        # Resample the HTF data to the LTF's frequency and forward-fill.
        # This means for any 5-minute candle, we will know the state of the last
        # closed 1-hour candle.
        aligned_htf_trend = htf_trend_data.resample(pandas_ltf_freq).ffill()

        # Join the aligned HTF data with the LTF data.
        aligned_df = df_ltf.join(aligned_htf_trend)
        
        return None, aligned_df

    def get_entry_signal(self, prev_row, current_row):
        """
        Determines the entry signal based on the aligned multi-timeframe data.
        """
        # --- Trend Conditions (from HTF) ---
        trend_up = current_row['htf_close'] > current_row[self.ema_col]
        trend_down = current_row['htf_close'] < current_row[self.ema_col]

        # --- Entry Conditions (from LTF) ---
        # Stoch RSI Bullish Crossover
        stoch_bullish_cross = (prev_row[self.stoch_k_col] <= prev_row[self.stoch_d_col] and
                               current_row[self.stoch_k_col] > current_row[self.stoch_d_col])
        
        # Stoch RSI Bearish Crossover
        stoch_bearish_cross = (prev_row[self.stoch_k_col] >= prev_row[self.stoch_d_col] and
                               current_row[self.stoch_k_col] < current_row[self.stoch_d_col])

        # Oversold/Overbought state
        is_oversold = current_row[self.stoch_k_col] < self.stoch_oversold
        is_overbought = current_row[self.stoch_k_col] > self.stoch_overbought

        # --- Final Signal ---
        # Long Entry: HTF is in an uptrend AND LTF has a bullish cross in the oversold zone.
        if trend_up and stoch_bullish_cross and is_oversold:
            return 'LONG'

        # Short Entry: HTF is in a downtrend AND LTF has a bearish cross in the overbought zone.
        if trend_down and stoch_bearish_cross and is_overbought:
            return 'SHORT'

        return None
