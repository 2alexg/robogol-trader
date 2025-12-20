# strategies/physics_strategy.py
from .ml_strategy import MLStrategy
import numpy as np
import pandas as pd

# --- HELPER FUNCTIONS ---
def wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series, window):
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    return wma((2 * wma(series, half_length)) - wma(series, window), sqrt_length)

class PhysicsStrategy(MLStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.features = ['velocity', 'acceleration', 'rsi', 'z_score', 'pct_b', 'atr_pct', 'adx']
        
        # Load ATR multipliers from config, defaulting to 2.0 to match massive_backtest_engine.py
        self.atr_sl_multiplier = float(params.get('atr_sl_multiplier', 2.0))
        self.atr_tp_multiplier = float(params.get('atr_tp_multiplier', 2.0))

    def add_features(self, df):
        df = df.copy()
        
        # 1. PHYSICS
        df['hma_50'] = hma(df['close'], 50)
        df['velocity'] = df['hma_50'].diff()
        df['acceleration'] = df['velocity'].diff()
        
        # 2. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. STATS
        df['ma_50'] = df['close'].rolling(50).mean()
        df['std_50'] = df['close'].rolling(50).std()
        df['std_50'] = df['std_50'].replace(0, np.nan)
        df['z_score'] = (df['close'] - df['ma_50']) / df['std_50']
        
        df['bb_up'] = df['ma_50'] + 2*df['std_50']
        df['bb_low'] = df['ma_50'] - 2*df['std_50']
        bb_range = df['bb_up'] - df['bb_low']
        bb_range = bb_range.replace(0, np.nan)
        df['pct_b'] = (df['close'] - df['bb_low']) / bb_range
        
        # 4. VOLATILITY (ATR) - CRITICAL
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        # Using rolling 20 period to match original file
        df['atr'] = df['tr'].rolling(20).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # 5. ADX (Included in Physics V2)
        period = 14
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['ndm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        df['tr_smooth'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
        df['pdm_smooth'] = df['pdm'].ewm(alpha=1/period, adjust=False).mean()
        df['ndm_smooth'] = df['ndm'].ewm(alpha=1/period, adjust=False).mean()
        df['pdi'] = 100 * (df['pdm_smooth'] / df['tr_smooth'])
        df['ndi'] = 100 * (df['ndm_smooth'] / df['tr_smooth'])
        df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])
        df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()

        df.dropna(inplace=True)
        return df

    def calculate_exit_prices(self, entry_price, signal, current_row):
        """
        Calculates Stop Loss and Take Profit based on the ATR value
        from the current row, using the multipliers from config.
        """
        atr = current_row.get('atr', 0)
        
        # Fallback if ATR is missing or zero (should be rare if add_features ran)
        if atr <= 0:
            # Fallback to a small percentage if ATR fails, or raise error
            atr = entry_price * 0.01 

        if signal == 'LONG':
            stop_loss = entry_price - (atr * self.atr_sl_multiplier)
            take_profit = entry_price + (atr * self.atr_tp_multiplier)
        elif signal == 'SHORT':
            stop_loss = entry_price + (atr * self.atr_sl_multiplier)
            take_profit = entry_price - (atr * self.atr_tp_multiplier)
        else:
            return None, None
            
        return stop_loss, take_profit