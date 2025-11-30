# optimizer_ga.py
#
# Description:
# A Genetic Algorithm (GA) based optimization engine.
# This version delegates exit price calculations to the strategy class,
# enabling support for complex exit logic like ATR-based stops.
#
# Author: Gemini
# Date: 2025-11-21 (Strategy-Aware Exits)

import random
import numpy as np
import pandas as pd
import time
import importlib.util
import re
import argparse
import sys

from deap import base, creator, tools, algorithms
import config
from data_manager import DataManager

# --- GA Configuration ---
POPULATION_SIZE = 50
GENERATIONS = 20
CXPB = 0.6
MUTPB = 0.3
MUTATION_INDPB = 0.2

# -- Static Trade Parameters --
INITIAL_CAPITAL = 10000
TRADE_SIZE_USD = 1000
FEE_PERCENT = 0.001

# --- Helper Functions ---
def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def load_strategy_class(strategy_name):
    try:
        module_name = camel_to_snake(strategy_name)
        module_path = f"strategies.{module_name}"
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, strategy_name)
        return strategy_class
    except (ImportError, AttributeError) as e:
        print(f"\n--- ERROR ---", file=sys.stderr)
        print(f"Error loading strategy: {strategy_name}", file=sys.stderr)
        print(f"Original error: {e}", file=sys.stderr)
        sys.exit(1)

def load_config_from_file(file_path):
    try:
        spec = importlib.util.spec_from_file_location("optimization_config", file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.GA_OPTIMIZATION_CONFIG
    except (FileNotFoundError, AttributeError) as e:
        print(f"\n--- ERROR ---", file=sys.stderr)
        print(f"Could not load configuration from '{file_path}'.", file=sys.stderr)
        sys.exit(1)


# --- Backtesting Core (Fitness Function) ---
def run_backtest(individual, strategy_config, data_manager):
    param_keys = strategy_config['parameters'].keys()
    params = dict(zip(param_keys, individual))

    # Note: We removed the hardcoded 'take_profit_percent' calculation here.
    # The strategy itself handles its parameters.

    strategy_class = strategy_config['strategy_class']

    try:
        params['symbol'] = strategy_config['symbol']
        
        high_timeframe = strategy_config.get('high_timeframe')
        
        if high_timeframe:
            df_htf = data_manager.get_data(params['symbol'], high_timeframe)
            df_ltf = data_manager.get_data(params['symbol'], strategy_config['timeframe'])
            if df_htf is None or df_htf.empty or df_ltf is None or df_ltf.empty: return (-99998,)
        elif strategy_class.is_multi_timeframe:
            params['htf'] = strategy_config['htf']
            params['ltf'] = strategy_config['ltf']
            df_htf = data_manager.get_data(params['symbol'], params['htf'])
            df_ltf = data_manager.get_data(params['symbol'], params['ltf'])
            if df_htf is None or df_htf.empty or df_ltf is None or df_ltf.empty: return (-99998,)
        else:
            params['timeframe'] = strategy_config['timeframe']
            df_ltf = data_manager.get_data(params['symbol'], params['timeframe'])
            if df_ltf is None or df_ltf.empty: return (-99998,)
            df_htf = None
        
        capital = INITIAL_CAPITAL
        position = None
        entry_price = 0
        sl_price, tp_price = 0, 0
        
        strategy = strategy_class(params)
        _, aligned_df = strategy.calculate_indicators(df_htf, df_ltf)
        
        if aligned_df is None: return (-99996,)
        aligned_df.dropna(inplace=True)
        if aligned_df.empty: return (-99995,)

        for i in range(1, len(aligned_df)):
            prev_row, current_row = aligned_df.iloc[i-1], aligned_df.iloc[i]
            
            # --- Exit Logic (Delegate to Strategy) ---
            if position:
                pnl, exit_reason = 0, None
                
                if position == 'LONG':
                    if current_row['low'] <= sl_price:
                        exit_reason, pnl = "SL", sl_price - entry_price
                    elif current_row['high'] >= tp_price:
                        exit_reason, pnl = "TP", tp_price - entry_price
                elif position == 'SHORT':
                    if current_row['high'] >= sl_price:
                        exit_reason, pnl = "SL", entry_price - sl_price
                    elif current_row['low'] <= tp_price:
                        exit_reason, pnl = "TP", entry_price - tp_price
                
                if exit_reason:
                    # Simple PnL calculation for optimization speed
                    capital += ((pnl / entry_price) * TRADE_SIZE_USD) - (TRADE_SIZE_USD * FEE_PERCENT * 2)
                    position = None

            # --- Entry Logic ---
            if not position:
                entry_signal = strategy.get_entry_signal(prev_row, current_row)
                if entry_signal:
                    position = entry_signal
                    entry_price = current_row['close']
                    
                    # --- FIX: Ask strategy for exit prices ---
                    sl_price, tp_price = strategy.calculate_exit_prices(entry_price, position, current_row)
        
        final_pnl = capital - INITIAL_CAPITAL
        return (final_pnl,)

    except Exception as e:
        return (-99997,)

# --- Full Statistics Backtest ---
def run_backtest_with_full_stats(params, strategy_config, data_manager):
    # Note: Removed hardcoded TP calculation here too.
    strategy_class = strategy_config['strategy_class']
    
    high_timeframe = strategy_config.get('high_timeframe')
    df_htf = None
    df_ltf = None
    
    if high_timeframe:
        df_htf = data_manager.get_data(params['symbol'], high_timeframe)
        df_ltf = data_manager.get_data(params['symbol'], strategy_config['timeframe'])
    elif strategy_class.is_multi_timeframe:
        df_htf = data_manager.get_data(params['symbol'], params['htf'])
        df_ltf = data_manager.get_data(params['symbol'], params['ltf'])
    else:
        df_ltf = data_manager.get_data(params['symbol'], params['timeframe'])
        df_htf = None

    capital = INITIAL_CAPITAL
    position = None
    entry_price = 0
    sl_price, tp_price = 0, 0
    
    strategy = strategy_class(params)
    _, aligned_df = strategy.calculate_indicators(df_htf, df_ltf)
    
    if aligned_df is None:
        return { "Final PnL": 0, "Total Trades": 0, "Win Rate (%)": 0, "Max Drawdown (%)": 0, "Profit Factor": 0, "Sharpe Ratio": 0 }
        
    aligned_df.dropna(inplace=True)

    equity_curve = [INITIAL_CAPITAL]
    trades = []
    
    for i in range(1, len(aligned_df)):
        prev_row, current_row = aligned_df.iloc[i-1], aligned_df.iloc[i]
        
        # --- Exit Logic ---
        if position:
            pnl, exit_reason = 0, None
            
            if position == 'LONG':
                if current_row['low'] <= sl_price:
                    exit_reason, pnl = "SL", sl_price - entry_price
                elif current_row['high'] >= tp_price:
                    exit_reason, pnl = "TP", tp_price - entry_price
            elif position == 'SHORT':
                if current_row['high'] >= sl_price:
                    exit_reason, pnl = "SL", entry_price - sl_price
                elif current_row['low'] <= tp_price:
                    exit_reason, pnl = "TP", entry_price - tp_price
            
            if exit_reason:
                net_pnl = ((pnl / entry_price) * TRADE_SIZE_USD) - (TRADE_SIZE_USD * FEE_PERCENT * 2)
                capital += net_pnl
                trades.append(net_pnl)
                position = None
        
        # --- Entry Logic ---
        if not position:
            entry_signal = strategy.get_entry_signal(prev_row, current_row)
            if entry_signal:
                position = entry_signal
                entry_price = current_row['close']
                # --- FIX: Ask strategy for exit prices ---
                sl_price, tp_price = strategy.calculate_exit_prices(entry_price, position, current_row)
        
        equity_curve.append(capital)

    final_pnl = capital - INITIAL_CAPITAL
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    returns = equity_series.pct_change().dropna()
    
    tf_str = params.get('ltf', params.get('timeframe', '1d'))
    periods_per_day = {'m': 1440, 'h': 24, 'd': 1}.get(tf_str[-1], 1)
    if tf_str[:-1].isdigit():
        if tf_str[-1] in 'mh': periods_per_day /= int(tf_str[:-1])
    annualization_factor = 252 * periods_per_day
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(annualization_factor) if returns.std() > 0 else 0

    return {
        "Final PnL": final_pnl,
        "Total Trades": len(trades),
        "Win Rate (%)": (sum(1 for t in trades if t > 0) / len(trades) * 100) if trades else 0,
        "Max Drawdown (%)": max_drawdown * 100,
        "Profit Factor": profit_factor,
        "Sharpe Ratio": sharpe_ratio
    }

# --- Main Optimizer Logic ---
def run_optimization(strategy_config):
    start_time = time.time()
    
    strategy_name = strategy_config['strategy_name']
    strategy_class = load_strategy_class(strategy_name)
    strategy_config['strategy_class'] = strategy_class
    
    param_definitions = strategy_config['parameters']

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    attributes = []
    param_keys = list(param_definitions.keys())
    
    for param, definition in param_definitions.items():
        if definition['type'] == 'int': toolbox.register(f"attr_{param}", random.randrange, definition['min'], definition['max'], definition['step'])
        elif definition['type'] == 'float': toolbox.register(f"attr_{param}", random.uniform, definition['min'], definition['max'])
        elif definition['type'] == 'categorical': toolbox.register(f"attr_{param}", random.choice, definition['choices'])
        attributes.append(getattr(toolbox, f"attr_{param}"))

    toolbox.register("individual", tools.initCycle, creator.Individual, tuple(attributes), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def custom_mutate(individual):
        for i, (param_key, definition) in enumerate(param_definitions.items()):
            if random.random() < MUTATION_INDPB:
                if definition['type'] == 'int': individual[i] = random.randrange(definition['min'], definition['max'], definition['step'])
                elif definition['type'] == 'float': individual[i] = random.uniform(definition['min'], definition['max'])
                elif definition['type'] == 'categorical' and len(definition['choices']) > 1: individual[i] = random.choice(definition['choices'])
        return individual,

    toolbox.register("mutate", custom_mutate)
    toolbox.register("mate", tools.cxTwoPoint)
    
    required_tfs = set()
    if strategy_config.get('high_timeframe'):
        required_tfs.add(strategy_config['high_timeframe'])
    if strategy_config.get('timeframe'):
        required_tfs.add(strategy_config['timeframe'])
    # Legacy support
    if strategy_config.get('htf'):
        required_tfs.add(strategy_config['htf'])
    if strategy_config.get('ltf'):
        required_tfs.add(strategy_config['ltf'])
        
    data_manager = DataManager([strategy_config['symbol']], list(required_tfs))
    
    toolbox.register("evaluate", run_backtest, strategy_config=strategy_config, data_manager=data_manager)
    toolbox.register("select", tools.selTournament, tournsize=3)

    print(f"--- Starting GA Optimization for '{strategy_name}' on '{strategy_config['symbol']}' ---")
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean); stats.register("std", np.std); stats.register("min", np.min); stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print("\n--- Optimization Complete ---")
    
    if not hof:
        print("No valid individual found.")
        return

    best_individual = hof[0]
    best_params_decoded = dict(zip(param_keys, best_individual))

    print("\nBest Individual Found:")
    for key, val in best_params_decoded.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    best_params_decoded['symbol'] = strategy_config['symbol']
    if strategy_config.get('high_timeframe'):
        best_params_decoded['high_timeframe'] = strategy_config['high_timeframe']
    if strategy_config.get('timeframe'):
        best_params_decoded['timeframe'] = strategy_config['timeframe']
    if strategy_config.get('htf'):
        best_params_decoded['htf'] = strategy_config['htf']
    if strategy_config.get('ltf'):
        best_params_decoded['ltf'] = strategy_config['ltf']

    print("\n--- Detailed Performance Report for Best Individual ---")
    full_stats = run_backtest_with_full_stats(best_params_decoded, strategy_config, data_manager)
    for key, value in full_stats.items():
        print(f"{key:<20}: {value:,.2f}")
    
    end_time = time.time()
    print(f"\nTotal optimization time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm Strategy Optimizer")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    ga_config = load_config_from_file(args.config)
    run_optimization(ga_config)
