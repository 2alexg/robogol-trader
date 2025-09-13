# optimizer.py
#
# Description:
# A multi-strategy optimization engine that runs backtests with explicitly
# defined parameter sets to find the most profitable configurations.
#
# Author: Gemini
# Date: 2025-07-29 (v7 - Fixed timeframe parsing bug)

import pandas as pd
import time
import config
from data_manager import DataManager
from strategies import IchimokuADXStrategy, HeikinAshiMACDStrategy, HeikinAshiMACDEMAStrategy

# --- Optimization Configuration ---
# We now define a list of explicit parameter sets for each strategy.
# This gives us full control over the combinations we want to test.

OPTIMIZATION_CONFIG = [
    # --- Suite 1: Test the HeikinAshiMACDEMA strategy with specific SL/TP ratios ---
    {
        'strategy_class': HeikinAshiMACDEMAStrategy,
        'parameter_sets': [
            # Set 1: 2:1 Reward/Risk Ratio
            {'timeframe': '4h', 'ema_period': 200, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'stop_loss_percent': 0.05, 'take_profit_percent': 0.10},
            # Set 2: 3:1 Reward/Risk Ratio
            {'timeframe': '4h', 'ema_period': 200, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'stop_loss_percent': 0.05, 'take_profit_percent': 0.15},
            # Set 3: Tighter SL with 2:1 Ratio
            {'timeframe': '4h', 'ema_period': 200, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'stop_loss_percent': 0.04, 'take_profit_percent': 0.08},
            # Set 4: 1h 2:1 Reward/Risk Ratio
            {'timeframe': '1h', 'ema_period': 200, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'stop_loss_percent': 0.03, 'take_profit_percent': 0.06},
            # Set 2: 1h 3:1 Reward/Risk Ratio
            {'timeframe': '1h', 'ema_period': 200, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'stop_loss_percent': 0.03, 'take_profit_percent': 0.09},
            # Set 3: 1h Tighter SL with 2:1 Ratio
            {'timeframe': '1h', 'ema_period': 200, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'stop_loss_percent': 0.02, 'take_profit_percent': 0.04},
        ]
    },
    # --- Suite 2: Test the IchimokuADX strategy with different ADX settings ---
    {
        'strategy_class': IchimokuADXStrategy,
        'parameter_sets': [
            # Set 1: Standard ADX
            {'htf': '1d', 'ltf': '1h', 'adx_period': 14, 'adx_threshold': 25, 'stop_loss_percent': 0.03, 'take_profit_percent': 0.06},
            # Set 2: More sensitive ADX
            {'htf': '1d', 'ltf': '1h', 'adx_period': 10, 'adx_threshold': 20, 'stop_loss_percent': 0.03, 'take_profit_percent': 0.06},
        ]
    }
]

# -- Static Trade Parameters --
INITIAL_CAPITAL = 10000
TRADE_SIZE_USD = 1000
FEE_PERCENT = 0.001
SYMBOL = 'BTC/USDT' # Common symbol for all tests for now

# --- Backtesting Core ---

def run_single_backtest(df_htf, df_ltf, params, strategy_class):
    """
    Runs a single backtest with a given set of parameters and a strategy object.
    """
    capital = INITIAL_CAPITAL
    trades = []
    position = None

    strategy = strategy_class(params)
    df_htf, df_ltf = strategy.calculate_indicators(df_htf, df_ltf)
    
    df_htf.dropna(inplace=True) if df_htf is not None else None
    df_ltf.dropna(inplace=True)

    if strategy.is_multi_timeframe:
        pandas_ltf_freq = params['ltf'].replace('m', 'min')
        htf_trend = df_htf[[strategy.adx_col, strategy.dmp_col, strategy.dmn_col]].resample(pandas_ltf_freq).ffill()
        aligned_df = df_ltf.join(htf_trend).dropna()
    else:
        aligned_df = df_ltf

    if aligned_df.empty:
        return {'pnl': 0, 'trades': 0, 'win_rate': 0}

    for i in range(1, len(aligned_df)):
        prev_row, current_row = aligned_df.iloc[i-1], aligned_df.iloc[i]

        if position: # Exit logic
            pnl, exit_reason = 0, None
            stop_loss = params['stop_loss_percent']
            take_profit = params['take_profit_percent']
            if position == 'LONG':
                if current_row['low'] <= entry_price * (1 - stop_loss):
                    exit_reason = "Stop Loss"
                    pnl = (entry_price * (1 - stop_loss)) - entry_price
                elif current_row['high'] >= entry_price * (1 + take_profit):
                    exit_reason = "Take Profit"
                    pnl = (entry_price * (1 + take_profit)) - entry_price
            elif position == 'SHORT':
                if current_row['high'] >= entry_price * (1 + stop_loss):
                    exit_reason = "Stop Loss"
                    pnl = entry_price - (entry_price * (1 + stop_loss))
                elif current_row['low'] <= entry_price * (1 - take_profit):
                    exit_reason = "Take Profit"
                    pnl = entry_price - (entry_price * (1 - take_profit))
            
            if exit_reason:
                net_pnl = ((pnl / entry_price) * TRADE_SIZE_USD) - (TRADE_SIZE_USD * FEE_PERCENT * 2)
                capital += net_pnl
                trades.append({'pnl': net_pnl})
                position = None

        if not position: # Entry logic
            entry_signal = strategy.get_entry_signal(prev_row, current_row)
            if entry_signal:
                position, entry_price = entry_signal, current_row['close']

    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    return {'pnl': capital - INITIAL_CAPITAL, 'trades': total_trades, 'win_rate': win_rate}

# --- Main Optimizer Logic ---

def main():
    start_time = time.time()
    
    # --- Collect all parameter sets and required data ---
    all_tests = []
    all_timeframes = set()

    for suite in OPTIMIZATION_CONFIG:
        strategy_class = suite['strategy_class']
        for param_set in suite['parameter_sets']:
            test = param_set.copy()
            test['strategy_class'] = strategy_class
            all_tests.append(test)
            
            # --- FIX: Correctly add timeframe strings to the set ---
            if 'htf' in param_set:
                all_timeframes.add(param_set['htf'])
            if 'ltf' in param_set:
                all_timeframes.add(param_set['ltf'])
            if 'timeframe' in param_set:
                all_timeframes.add(param_set['timeframe'])

    print(f"Generated {len(all_tests)} specific tests to run.")
    
    data_manager = DataManager([SYMBOL], all_timeframes)
    results = []

    for i, params in enumerate(all_tests):
        strategy_class = params['strategy_class']
        param_display = {k: v for k, v in params.items() if k != 'strategy_class'}
        param_display['strategy'] = strategy_class.__name__
        print(f"\n--- Running Test {i+1}/{len(all_tests)} ---")
        print(f"Params: {param_display}")
        
        try:
            if strategy_class.is_multi_timeframe:
                df_htf = data_manager.get_data(SYMBOL, params['htf'])
                df_ltf = data_manager.get_data(SYMBOL, params['ltf'])
            else:
                df_ltf = data_manager.get_data(SYMBOL, params['timeframe'])
                df_htf = None

            result = run_single_backtest(df_htf, df_ltf, params, strategy_class)
            
            result_summary = param_display.copy()
            result_summary.update(result)
            results.append(result_summary)
            print(f"Result: PnL=${result['pnl']:.2f}, Trades={result['trades']}, Win Rate={result['win_rate']:.2f}%")
        except Exception as e:
            print(f"An error occurred during this run: {e}")

    print("\n--- Optimization Complete ---")
    if not results:
        print("No results to display.")
        return

    results_df = pd.DataFrame(results)
    sorted_results = results_df.sort_values(by='pnl', ascending=False)
    
    print("Top Performing Parameter Sets:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(sorted_results.head(10).to_string(index=False))
    
    end_time = time.time()
    print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
