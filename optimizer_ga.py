# optimizer_ga.py
#
# Description:
# A Rolling-Window Genetic Algorithm Optimizer.
# It iterates through history, optimizing strategy parameters on a rolling
# "Training Window" (10k candles) and simulating the best parameters on a
# "Testing Window" (1k candles).
#
# Logs trades to MongoDB in a format compatible with generate_smart_whitelist.py.
#
# Usage:
# python optimizer_ga.py --config optimization_config_macd.py
#
# Author: Gemini
# Date: 2025-12-23

import random
import numpy as np
import pandas as pd
import time
import importlib.util
import re
import argparse
import sys
import datetime
import uuid
from pymongo import MongoClient
from deap import base, creator, tools, algorithms

# --- Project Imports ---
try:
    import config
    # Default Config Fallbacks
    MONGO_URI = getattr(config, 'OPTIM_MONGO_URI', "mongodb://localhost:27017/")
    # DATA_DB_NAME is used implicitly by DataManager via config.py
    SYMBOLS = getattr(config, 'SYMBOLS', ["BTC/USDT"])
    TIMEFRAMES = getattr(config, 'TIMEFRAMES', ["5m"])
except ImportError:
    print("WARNING: Could not import 'config.py'. Using defaults.")
    MONGO_URI = "mongodb://localhost:27017/"
    SYMBOLS = ["BTC/USDT"]
    TIMEFRAMES = ["5m"]

from data_manager import DataManager

# --- CONSTANTS ---
LOG_DB_NAME = "backtest_logs" # Separate DB for simulation results
TRAIN_LEN = 10000      # Size of the optimization window
TEST_LEN = 1000        # Size of the forward test/simulation window
WARMUP_LEN = 500       # Size of the warmup period for indicators
WINDOW_SIZE = TRAIN_LEN
MIN_TRAIN_PRECISION = 50.0  # Min Win Rate (%) in training to justify trading the test window

# GA Constants
POPULATION_SIZE = 40   # Reduced slightly for speed in rolling loop
GENERATIONS = 10
CXPB = 0.6
MUTPB = 0.3
MUTATION_INDPB = 0.2

# Trade Constants
INITIAL_CAPITAL = 10000
TRADE_SIZE_USD = 1000
FEE_PERCENT = 0.00075  # 0.075%

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- HELPER FUNCTIONS ---

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
        print(f"Error loading strategy '{strategy_name}': {e}", file=sys.stderr)
        sys.exit(1)

def load_config_from_file(file_path):
    try:
        spec = importlib.util.spec_from_file_location("optimization_config", file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.GA_OPTIMIZATION_CONFIG
    except Exception as e:
        print(f"Error loading config file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

# --- DATABASE LOGGING ---

class TradeLogger:
    def __init__(self, uri, db_name, strategy_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name] # Explicitly uses the log DB name
        self.trades = self.db['backtest_trades']
        self.runs = self.db['backtest_runs']

        # Indexes
        self.trades.create_index("run_id")
        self.trades.create_index("entry_time")

        # Log Run Start
        self.runs.insert_one({
            "run_id": RUN_ID,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "strategy": strategy_name,
            "config": {
                "train_len": TRAIN_LEN,
                "test_len": TEST_LEN,
                "warmup_len": WARMUP_LEN,
                "symbols": SYMBOLS,
                "timeframes": TIMEFRAMES
            },
            "status": "running"
        })
        print(f"--- LOGGING STARTED (Run ID: {RUN_ID}) ---")
        print(f"    Target DB: '{db_name}'")

    def log_trades(self, trade_list):
        if trade_list:
            self.trades.insert_many(trade_list)

    def finish_run(self, total_pnl):
        self.runs.update_one(
            {"run_id": RUN_ID},
            {"$set": {"status": "completed", "total_pnl": total_pnl, "end_time": datetime.datetime.now(datetime.timezone.utc)}}
        )

# --- BACKTESTING ENGINES ---

def run_strategy_logic(df, strategy_class, params):
    """
    Instantiates strategy and calculates indicators/signals.
    Returns the dataframe with signal columns or None on failure.
    """
    # Create Strategy Instance
    try:
        strategy = strategy_class(params)
    except Exception as e:
        # print(f"Strategy Init Error: {e}")
        return None, None

    # We assume single timeframe optimization for simplicity in this loop,
    # or HTF data must be passed if required.
    # For this implementation, we use the provided DF as the primary LTF data.
    # If the strategy requires HTF, it likely won't work well in this generic
    # rolling window without fetching HTF specifically.
    # We will pass None for HTF unless we engineer complex data merging here.

    try:
        # Important: Pass a COPY to avoid polluting the original dataframe in the loop
        _, df_processed = strategy.calculate_indicators(None, df.copy())
    except Exception as e:
        # print(f"Indicator Error: {e}")
        return None, None

    if df_processed is None or df_processed.empty:
        return None, None

    df_processed.dropna(inplace=True)
    return strategy, df_processed

def evaluate_fitness(individual, strategy_class, df_train, param_keys, min_date):
    """
    Fitness function for GA. Returns PnL of the Training Window.
    """
    params = dict(zip(param_keys, individual))

    strategy, df = run_strategy_logic(df_train, strategy_class, params)

    # Filter out warmup data so we evaluate only on the intended window
    if df is not None and not df.empty:
        df = df[df.index >= min_date]

    if strategy is None or df is None or df.empty:
        return (-9999.0,) # High penalty

    capital = INITIAL_CAPITAL
    position = None
    entry_price = 0
    sl_price, tp_price = 0, 0

    # Fast Loop for PnL only
    # Note: To speed up GA, we strictly calculate PnL here.
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # Exit
        if position:
            pnl = 0
            exit_hit = False
            if position == 'LONG':
                if current_row['low'] <= sl_price:
                    pnl = sl_price - entry_price; exit_hit = True
                elif current_row['high'] >= tp_price:
                    pnl = tp_price - entry_price; exit_hit = True
            elif position == 'SHORT':
                if current_row['high'] >= sl_price:
                    pnl = entry_price - sl_price; exit_hit = True
                elif current_row['low'] <= tp_price:
                    pnl = entry_price - tp_price; exit_hit = True

            if exit_hit:
                capital += ((pnl / entry_price) * TRADE_SIZE_USD) - (TRADE_SIZE_USD * FEE_PERCENT * 2)
                position = None

        # Entry
        if not position:
            sig = strategy.get_entry_signal(prev_row, current_row)
            if sig:
                position = sig
                entry_price = current_row['close']
                sl_price, tp_price = strategy.calculate_exit_prices(entry_price, position, current_row)

    net_pnl = capital - INITIAL_CAPITAL
    return (net_pnl,)

def simulate_and_get_trades(best_params, strategy_class, df_test, symbol, timeframe, is_training=False, min_date=None):
    """
    Runs the simulation on the Test Window (or Train Window if is_training=True)
    Returns:
       if is_training=True:  (WinRate, PnL)
       if is_training=False: List of trade dictionaries for logging
    """
    strategy, df = run_strategy_logic(df_test, strategy_class, best_params)

    # Filter out warmup data so we evaluate only on the intended window
    if df is not None and not df.empty and min_date is not None:
        df = df[df.index >= min_date]

    if strategy is None or df is None or df.empty:
        return (0, 0) if is_training else []

    trades_log = []
    capital = INITIAL_CAPITAL
    position = None
    entry_price = 0
    sl_price, tp_price = 0, 0
    entry_time = None

    # Store trades for stats
    trade_pnls = []

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # Exit Logic
        if position:
            pnl = 0
            status = None
            exit_price_exec = 0

            if position == 'LONG':
                if current_row['low'] <= sl_price:
                    exit_price_exec = sl_price
                    pnl = sl_price - entry_price
                    status = 'loss'
                elif current_row['high'] >= tp_price:
                    exit_price_exec = tp_price
                    pnl = tp_price - entry_price
                    status = 'win'
            elif position == 'SHORT':
                if current_row['high'] >= sl_price:
                    exit_price_exec = sl_price
                    pnl = entry_price - sl_price
                    status = 'loss'
                elif current_row['low'] <= tp_price:
                    exit_price_exec = tp_price
                    pnl = entry_price - tp_price
                    status = 'win'

            if status:
                net_pnl = ((pnl / entry_price) * TRADE_SIZE_USD) - (TRADE_SIZE_USD * FEE_PERCENT * 2)
                capital += net_pnl
                trade_pnls.append(net_pnl)

                if not is_training:
                    trades_log.append({
                        "run_id": RUN_ID,
                        "strategy": strategy_class.__name__,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "side": position,
                        "entry_time": entry_time,
                        "exit_time": current_row.name, # Timestamp index
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price_exec),
                        "pnl_net": float(net_pnl),
                        "status": status,
                        "optimized_params": best_params # Log the params that generated this trade
                    })

                position = None

        # Entry Logic
        if not position:
            sig = strategy.get_entry_signal(prev_row, current_row)
            if sig:
                position = sig
                entry_price = current_row['close']
                entry_time = current_row.name
                sl_price, tp_price = strategy.calculate_exit_prices(entry_price, position, current_row)

    # Return Logic
    if is_training:
        total_trades = len(trade_pnls)
        wins = len([p for p in trade_pnls if p > 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(trade_pnls)
        return win_rate, total_pnl
    else:
        return trades_log

# --- MAIN ROLLING WINDOW LOOP ---

def run_rolling_optimization(ga_config):
    # 1. Setup
    strategy_name = ga_config['strategy_name']
    strategy_class = load_strategy_class(strategy_name)
    param_definitions = ga_config['parameters']
    param_keys = list(param_definitions.keys())

    # Initialize Logger with the separate LOG_DB_NAME
    logger = TradeLogger(MONGO_URI, LOG_DB_NAME, strategy_name)

    # 2. Initialize DEAP Toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    attributes = []
    for param, definition in param_definitions.items():
        if definition['type'] == 'int':
            toolbox.register(f"attr_{param}", random.randrange, definition['min'], definition['max'], definition['step'])
        elif definition['type'] == 'float':
            toolbox.register(f"attr_{param}", random.uniform, definition['min'], definition['max'])
        elif definition['type'] == 'categorical':
            toolbox.register(f"attr_{param}", random.choice, definition['choices'])
        attributes.append(getattr(toolbox, f"attr_{param}"))

    toolbox.register("individual", tools.initCycle, creator.Individual, tuple(attributes), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def custom_mutate(individual):
        for i, (param_key, definition) in enumerate(param_definitions.items()):
            if random.random() < MUTATION_INDPB:
                if definition['type'] == 'int':
                    individual[i] = random.randrange(definition['min'], definition['max'], definition['step'])
                elif definition['type'] == 'float':
                    individual[i] = random.uniform(definition['min'], definition['max'])
                elif definition['type'] == 'categorical':
                    individual[i] = random.choice(definition['choices'])
        return individual,

    toolbox.register("mutate", custom_mutate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 3. Data Loading
    # We fetch ALL data needed for SYMBOLS + TIMEFRAMES from the Data DB (via config)
    data_manager = DataManager(SYMBOLS, TIMEFRAMES)

    total_portfolio_pnl = 0.0

    # 4. Main Loops
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"\n=== Processing {symbol} {timeframe} ===")

            # Get Full Data
            df_full = data_manager.get_data(symbol, timeframe)
            if df_full is None or len(df_full) < (TRAIN_LEN + TEST_LEN):
                print(f"  Not enough data for {symbol} {timeframe}. Skipping.")
                continue

            # Rolling Window Loop
            # We slide by TEST_LEN (1000 candles)
            max_start = len(df_full) - TRAIN_LEN - TEST_LEN

            for start_idx in range(0, max_start, TEST_LEN):
                train_end = start_idx + TRAIN_LEN
                test_end = train_end + TEST_LEN

                # Define Slices with Warmup
                # Train Slice: Include WARMUP_LEN candles before start_idx (if available)
                train_start_with_warmup = max(0, start_idx - WARMUP_LEN)
                df_train = df_full.iloc[train_start_with_warmup : train_end].copy()

                # Test Slice: Include WARMUP_LEN candles before train_end (start of test)
                test_start_with_warmup = max(0, train_end - WARMUP_LEN)
                df_test = df_full.iloc[test_start_with_warmup : test_end].copy()

                # Determine Cutoff Dates (The actual start of evaluation)
                # These timestamps mark where the real window begins, excluding warmup
                train_min_date = df_full.index[start_idx]
                test_min_date = df_full.index[train_end]

                # --- A. OPTIMIZATION PHASE (ON TRAIN WINDOW) ---
                # Register evaluate with the current df_train and the train_min_date cutoff
                toolbox.register("evaluate", evaluate_fitness, strategy_class=strategy_class,
                                 df_train=df_train, param_keys=param_keys, min_date=train_min_date)

                # Run GA
                pop = toolbox.population(n=POPULATION_SIZE)
                # Suppress detailed print per generation to keep logs clean
                algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS, verbose=False)

                # Get Best
                top_ind = tools.selBest(pop, 1)[0]
                best_params = dict(zip(param_keys, top_ind))

                # Validation: Check Precision/WinRate on Train Data
                # Note: We pass min_date so metrics are calculated only on the TRAIN_LEN window
                train_wr, train_pnl = simulate_and_get_trades(best_params, strategy_class, df_train, symbol, timeframe, is_training=True, min_date=train_min_date)

                current_date = df_test.index[0] # This might be inside warmup, but it's just for display label

                # --- B. SIMULATION PHASE (ON TEST WINDOW) ---
                if train_wr >= MIN_TRAIN_PRECISION:
                    print(f"  [{current_date}] Train WR: {train_wr:.1f}% | PnL: ${train_pnl:.0f} -> DEPLOYING params.")

                    # Simulate on Test Data
                    # Note: We pass min_date so trades are logged only for the TEST_LEN window
                    trades = simulate_and_get_trades(best_params, strategy_class, df_test, symbol, timeframe, is_training=False, min_date=test_min_date)

                    if trades:
                        logger.log_trades(trades)
                        window_pnl = sum(t['pnl_net'] for t in trades)
                        total_portfolio_pnl += window_pnl
                        print(f"     > Test Result: {len(trades)} trades, PnL: ${window_pnl:.2f}")
                    else:
                        print("     > No trades triggered in test window.")
                else:
                    print(f"  [{current_date}] Train WR: {train_wr:.1f}% (Low) -> SKIPPING deployment.")

    logger.finish_run(total_portfolio_pnl)
    print(f"\nDONE. Run ID: {RUN_ID}")
    print(f"Total Simulated PnL: ${total_portfolio_pnl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Window GA Optimizer")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to optimization config file (e.g., optimization_config_macd.py)")
    args = parser.parse_args()

    ga_config = load_config_from_file(args.config)

    # Clean up DEAP Creator classes if they exist from previous runs (in interactive envs)
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    run_rolling_optimization(ga_config)
