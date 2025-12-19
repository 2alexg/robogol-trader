import pandas as pd
import numpy as np
import datetime
import uuid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight
from pymongo import MongoClient
import sys
from strategies.physics_strategy import PhysicsStrategy
from strategies.vwap_strategy import VwapStrategy

# Map names to classes
STRATEGY_MAP = {
    "Physics": PhysicsStrategy,
    "Vwap": VwapStrategy
}

# --- CONFIGURATION ---
try:
    import config
    MONGO_URI = config.OPTIM_MONGO_URI
    DB_NAME = config.OPTIM_MONGO_DB_NAME
    # Overwrite symbols if you want to test ALL of them
    SYMBOLS = config.SYMBOLS 
    TIMEFRAMES = config.TIMEFRAMES 
except ImportError:
    print("Config not found, using defaults.")
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "crypto_data"
    SYMBOLS = ["ADA/USDT", "ETH/USDT", "SOL/USDT", "BTC/USDT", "BNB/USDT"]
    TIMEFRAMES = ['5m', '15m', '1h']

# --- TRADING SETTINGS ---
TRADE_SIZE = 1000.0
COMMISSION_RATE = 0.00075 # 0.075%
LOG_DB_NAME = "backtest_logs" # New DB for logs
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Unique ID for this run

# --- WINDOW CONFIGURATION ---
VAL_LEN = 3000     # Oldest data (Validation)
TRAIN_LEN = 7000   # Newest data (Training)
TEST_LEN = 1000    # Simulation Window
HORIZON = 100      # Max hold time
WINDOW_SIZE = VAL_LEN + TRAIN_LEN

# ==========================================
# 1. DATABASE & LOGGING
# ==========================================
class TradeLogger:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.trades = self.db['backtest_trades']
        self.runs = self.db['backtest_runs']
        
        # Create indexes for fast querying later
        self.trades.create_index("run_id")
        self.trades.create_index("entry_time")
        self.trades.create_index("symbol")
        
        # Log Run Start
        self.runs.insert_one({
            "run_id": RUN_ID,
            "timestamp": datetime.datetime.now(datetime.UTC),
            "config": {
                "train_len": TRAIN_LEN,
                "symbols": SYMBOLS,
                "timeframes": TIMEFRAMES
            },
            "status": "running"
        })
        print(f"--- LOGGING STARTED (Run ID: {RUN_ID}) ---")

    def log_trades(self, trade_list):
        if trade_list:
            self.trades.insert_many(trade_list)

    def finish_run(self, total_pnl):
        self.runs.update_one(
            {"run_id": RUN_ID},
            {"$set": {"status": "completed", "total_pnl": total_pnl, "end_time": datetime.datetime.now(datetime.UTC)}}
        )

def get_labels(df, side='short', sl_mult=2.0, tp_mult=2.0, horizon=100):
    labels = []
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr'].values
    length = len(df)
    
    for i in range(length - horizon):
        current_price = close_arr[i]
        current_atr = atr_arr[i]
        
        if side == 'short':
            stop_loss = current_price + (current_atr * sl_mult)
            take_profit = current_price - (current_atr * tp_mult)
            sl_hit = high_arr[i+1 : i+1+horizon] >= stop_loss
            tp_hit = low_arr[i+1 : i+1+horizon] <= take_profit
        else:
            stop_loss = current_price - (current_atr * sl_mult)
            take_profit = current_price + (current_atr * tp_mult)
            sl_hit = low_arr[i+1 : i+1+horizon] <= stop_loss
            tp_hit = high_arr[i+1 : i+1+horizon] >= take_profit
            
        has_sl = sl_hit.any()
        has_tp = tp_hit.any()
        first_sl = np.argmax(sl_hit) if has_sl else 9999
        first_tp = np.argmax(tp_hit) if has_tp else 9999
        
        if has_tp and first_tp < first_sl:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)

# ==========================================
# 2. TRAINING ENGINE
# ==========================================
def train_and_get_threshold(X_train_full, y_train_full, target_precision=0.70):
    # Split: Train on Newest 7000 (70%), Validate on Oldest 3000 (30%)
    X_val = X_train_full.iloc[:VAL_LEN]
    y_val = y_train_full[:VAL_LEN]
    
    X_train = X_train_full.iloc[VAL_LEN:]
    y_train = y_train_full[VAL_LEN:]
    
    if len(np.unique(y_train)) < 2: return None, 1.0, False

    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Check "Safety Valve" on Validation set
    probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    
    optimal_threshold = 1.0 
    found_good_model = False
    
    for p, t in zip(precisions[:-1], thresholds):
        if p >= target_precision:
            optimal_threshold = t
            found_good_model = True
            break
            
    return model, optimal_threshold, found_good_model

# ==========================================
# 3. SIMULATION & LOGGING ENGINE (THE NEW PART)
# ==========================================
def simulate_and_log(symbol, timeframe, preds, df_buffer, side, atr_mult=2.0):
    """
    Looks forward to find EXACT Exit Time for log.
    df_buffer: Contains [TEST_LEN + HORIZON] rows to allow looking ahead.
    """
    trades_to_log = []
    
    closes = df_buffer['close'].values
    highs = df_buffer['high'].values
    lows = df_buffer['low'].values
    atrs = df_buffer['atr'].values
    timestamps = df_buffer['timestamp'].values
    
    # Iterate only through the predictions (TEST_LEN)
    for i, pred in enumerate(preds):
        if pred == 1:
            entry_time = timestamps[i]
            entry_price = closes[i]
            volatility = atrs[i]
            
            # Define Targets
            if side == 'short':
                stop_loss = entry_price + (volatility * atr_mult)
                take_profit = entry_price - (volatility * atr_mult)
            else:
                stop_loss = entry_price - (volatility * atr_mult)
                take_profit = entry_price + (volatility * atr_mult)
            
            # Find Exit (Scan forward up to HORIZON)
            exit_time = None
            exit_price = entry_price # Default if timeout
            status = 'timeout'
            
            for j in range(1, HORIZON):
                if (i + j) >= len(highs): break # End of data
                
                curr_high = highs[i+j]
                curr_low = lows[i+j]
                curr_time = timestamps[i+j]
                
                # Check Hit
                sl_hit = False
                tp_hit = False
                
                if side == 'short':
                    if curr_high >= stop_loss: sl_hit = True
                    if curr_low <= take_profit: tp_hit = True
                else:
                    if curr_low <= stop_loss: sl_hit = True
                    if curr_high >= take_profit: tp_hit = True
                
                # Resolve Conflict (Assumption: TP hit first if both in same candle, or worst case SL)
                # To be conservative, let's assume SL hit first if both happen (Worst Case)
                if sl_hit:
                    exit_price = stop_loss
                    exit_time = curr_time
                    status = 'loss'
                    break
                elif tp_hit:
                    exit_price = take_profit
                    exit_time = curr_time
                    status = 'win'
                    break
            
            # If never hit, we exit at close of Horizon
            if exit_time is None:
                if (i + HORIZON) < len(closes):
                    exit_price = closes[i+HORIZON]
                    exit_time = timestamps[i+HORIZON]
                else:
                    exit_price = closes[-1]
                    exit_time = timestamps[-1]

            # Calculate Exact PnL
            pct_change = (exit_price - entry_price) / entry_price
            if side == 'short': pct_change = -pct_change
            
            gross_pnl = TRADE_SIZE * pct_change
            comm_cost = TRADE_SIZE * (COMMISSION_RATE * 2)
            net_pnl = gross_pnl - comm_cost
            
            # Construct Log Object
            trade_doc = {
                "run_id": RUN_ID,
                "symbol": symbol,
                "timeframe": timeframe,
                "side": side,
                "entry_time": pd.to_datetime(entry_time),
                "exit_time": pd.to_datetime(exit_time),
                "duration_minutes": (pd.to_datetime(exit_time) - pd.to_datetime(entry_time)).total_seconds() / 60,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl_net": float(net_pnl),
                "status": status
            }
            trades_to_log.append(trade_doc)
            
    return trades_to_log

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    logger = TradeLogger(MONGO_URI, LOG_DB_NAME)
    repo = MongoClient(MONGO_URI)[DB_NAME] # Direct access for loading data

    SELECTED_STRATEGY = "Physics" # Change this string to switch brains!
    StrategyClass = STRATEGY_MAP[SELECTED_STRATEGY]

    strategy_instance = StrategyClass({'model_path': None})

    features = ['velocity', 'acceleration', 'rsi', 'z_score', 'pct_b', 'atr_pct', 'adx', 'dist_vwap', 'rvol', 'chop_idx']
    total_portfolio_pnl = 0.0

    for timeframe in TIMEFRAMES:
        # LOGGING TIMESTAMP 1: Timeframe Level
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}] === TIMEFRAME: {timeframe} ===")
        
        for symbol in SYMBOLS:
            # LOGGING TIMESTAMP 2: Symbol Level
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Processing {symbol}...")
            
            # Load Data
            coll_name = f"{symbol.replace('/', '_')}_{timeframe}"
            cursor = repo[coll_name].find().sort("timestamp", 1)
            df = pd.DataFrame(list(cursor))
            if len(df) < (WINDOW_SIZE + TEST_LEN + HORIZON):
                continue
                
            # Cleanup & Features
            if 'metadata' in df.columns: del df['metadata']
            if '_id' in df.columns: del df['_id']
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
            
            # df = add_features(df)
            df = strategy_instance.add_features(df)
            features = strategy_instance.features # Get the exact feature list this model expects = strategy_instance.add_features(df)


            y_short = get_labels(df, side='short')
            y_long = get_labels(df, side='long')
            
            # Main Slide Loop
            max_start = len(df) - WINDOW_SIZE - TEST_LEN - HORIZON
            
            for start_idx in range(0, max_start, TEST_LEN):
                train_end = start_idx + WINDOW_SIZE
                test_end = train_end + TEST_LEN
                
                # Training Slice
                X_train_full = df.iloc[start_idx:train_end][features]
                y_s_full = y_short[start_idx:train_end]
                y_l_full = y_long[start_idx:train_end]
                
                # Simulation Slice (Inputs)
                X_sim = df.iloc[train_end:test_end][features]
                
                # Buffer Slice (For looking up Exit prices)
                # We need data from train_end up to test_end + HORIZON
                df_buffer = df.iloc[train_end : test_end + HORIZON]
                
                # Train & Predict
                model_s, t_s, active_s = train_and_get_threshold(X_train_full, y_s_full)
                preds_s = (model_s.predict_proba(X_sim)[:, 1] >= t_s).astype(int) if active_s else np.zeros(len(X_sim))
                
                model_l, t_l, active_l = train_and_get_threshold(X_train_full, y_l_full)
                preds_l = (model_l.predict_proba(X_sim)[:, 1] >= t_l).astype(int) if active_l else np.zeros(len(X_sim))
                
                # Simulate & Log
                trades_s = simulate_and_log(symbol, timeframe, preds_s, df_buffer, 'short')
                trades_l = simulate_and_log(symbol, timeframe, preds_l, df_buffer, 'long')
                
                # Batch Insert to Mongo
                all_trades = trades_s + trades_l
                if all_trades:
                    logger.log_trades(all_trades)
                    
                    # Accumulate PnL for print
                    week_pnl = sum(t['pnl_net'] for t in all_trades)
                    total_portfolio_pnl += week_pnl
                    # print(f"  Logged {len(all_trades)} trades. Week PnL: ${week_pnl:.2f}")

    logger.finish_run(total_portfolio_pnl)
    print(f"\nDONE. Run ID: {RUN_ID}")
    print(f"Total Portfolio PnL: ${total_portfolio_pnl:.2f}")
    print(f"Logs saved to MongoDB database: '{LOG_DB_NAME}', collection: 'backtest_trades'")

if __name__ == "__main__":
    main()
