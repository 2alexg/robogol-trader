# train_deployable_model.py
import pandas as pd
import numpy as np
import joblib
import argparse
from pymongo import MongoClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight

# Import your strategies
from strategies.physics_strategy import PhysicsStrategy
from strategies.vwap_strategy import VwapStrategy

# --- CONFIGURATION ---
try:
    import config
    MONGO_URI = config.OPTIM_MONGO_URI
    DB_NAME = config.OPTIM_MONGO_DB_NAME
except ImportError:
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "crypto_data"

STRATEGY_MAP = {
    "Physics": PhysicsStrategy,
    "Vwap": VwapStrategy
}

# --- 1. Labeling Logic (Same as Backtester) ---
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
    
    # Pad the end with 0s to match length
    return np.array(labels + [0] * horizon)

# --- 2. Training Function ---
def train_model(X, y, target_precision=0.70):
    weights = compute_sample_weight(class_weight='balanced', y=y)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y, sample_weight=weights)
    
    # Find Optimal Threshold
    probs = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, probs)
    
    optimal_threshold = 0.85 # Default fallback
    found_threshold = False
    
    for p, t in zip(precisions[:-1], thresholds):
        if p >= target_precision:
            optimal_threshold = t
            found_threshold = True
            break
    
    if not found_threshold:
        print(f"    Warning: Could not reach {target_precision} precision. Using best available.")
            
    return model, optimal_threshold

# --- 3. Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Train and save an ML model for trading.")
    parser.add_argument('--symbol', type=str, required=True, help="Symbol to train on (e.g., BTC/USDT)")
    parser.add_argument('--timeframe', type=str, required=True, help="Timeframe (e.g., 5m)")
    parser.add_argument('--strategy', type=str, default="Physics", choices=STRATEGY_MAP.keys(), help="Strategy class to use")
    parser.add_argument('--limit', type=int, default=10000, help="Number of recent candles to use for training (Default: 10000)")
    parser.add_argument('--output', type=str, default="model.pkl", help="Output filename for the pickle")
    args = parser.parse_args()

    print(f"--- Training {args.strategy} Model for {args.symbol} {args.timeframe} ---")
    print(f"--- Using latest {args.limit} candles ---")

    # 1. Load Data
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection_name = f"{args.symbol.replace('/', '_')}_{args.timeframe}"
    
    # We need to fetch slightly more than 'limit' to account for indicator warmup (e.g. +200 candles)
    fetch_limit = args.limit + 200
    
    # Sorting by timestamp DESCENDING (-1) to get the newest data first, then limit, then flip back
    cursor = db[collection_name].find().sort("timestamp", -1).limit(fetch_limit)
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("No data found. Please run data_ingestion_engine.py first.")
        return

    # Flip back to ascending order (Oldest -> Newest)
    df = df.iloc[::-1].reset_index(drop=True)

    # Clean Data
    if 'metadata' in df.columns: del df['metadata']
    if '_id' in df.columns: del df['_id']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']: 
        df[col] = df[col].astype(float)

    # 2. Add Features (Using the Strategy Class logic)
    StrategyClass = STRATEGY_MAP[args.strategy]
    strategy_instance = StrategyClass({'model_path': None}) 
    
    print("Generating Features...")
    df = strategy_instance.add_features(df)
    features = strategy_instance.features
    
    # After adding features, slice to the exact requested limit
    # This ensures we are training on exactly the last N clean candles
    if len(df) > args.limit:
        df = df.iloc[-args.limit:].copy()

    print(f"Training on {len(df)} candles. Date range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    # 3. Generate Targets
    print("Generating Labels...")
    y_short = get_labels(df, side='short')
    y_long = get_labels(df, side='long')

    # Remove the horizon rows (where we can't look forward)
    HORIZON = 100
    X = df.iloc[:-HORIZON][features]
    y_short = y_short[:-HORIZON]
    y_long = y_long[:-HORIZON]

    # 4. Train Models
    print("Training Short Model...")
    model_s, thresh_s = train_model(X, y_short)
    print(f"  > Done. Threshold: {thresh_s:.4f}")

    print("Training Long Model...")
    model_l, thresh_l = train_model(X, y_long)
    print(f"  > Done. Threshold: {thresh_l:.4f}")

    # 5. Save "Brain"
    brain = {
        'short_model': model_s,
        'long_model': model_l,
        'short_threshold': thresh_s,
        'long_threshold': thresh_l,
        'features': features,
        'strategy': args.strategy,
        'symbol': args.symbol,
        'timeframe': args.timeframe
    }
    
    joblib.dump(brain, args.output)
    print(f"\nSUCCESS: Model saved to '{args.output}'")

if __name__ == "__main__":
    main()