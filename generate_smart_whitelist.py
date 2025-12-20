import pandas as pd
from pymongo import MongoClient
import json
import os
import numpy as np

# --- CONFIGURATION ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "backtest_logs"

RUN_ID = "20251213_044243"  # Set to a specific ID (e.g., "20231027_153000") or None to analyze ALL trades

# Criteria for "Good" Models
MIN_PNL_WIN = 500.0     
MIN_WIN_RATE = 51.0   

# Criteria for "Perfect Losers" (To Reverse)
MAX_PNL_LOSS = -500.0      # Must lose at least this much
MAX_WIN_RATE_LOSS = 48.0   # Win rate must be consistently bad (<48%)
MIN_TRADES = 50            # Need sample size

def analyze_curve_smoothness(pnl_series):
    """
    Calculates R-squared of the PnL curve to check if it's 'constant'.
    Close to 1.0 = Very smooth/constant line.
    """
    if len(pnl_series) < 10: return 0
    y = pnl_series.cumsum()
    x = np.arange(len(y))
    # Correlation matrix
    correlation = np.corrcoef(x, y)[0, 1]
    return correlation**2 # R-squared

def generate_smart_whitelist():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    query = {}
    if RUN_ID:
        query['run_id'] = RUN_ID
        
    print("Fetching trades from MongoDB...")
    cursor = db.backtest_trades.find(query)
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("No trades found.")
        return

    # Sort for curve analysis
    df = df.sort_values('exit_time')

    # Group by Symbol + Timeframe
    pairs = df.groupby(['symbol', 'timeframe'])
    
    whitelist_config = {}
    
    print(f"\n{'PAIR':<20} | {'MODE':<8} | {'PnL':<10} | {'WR%':<6} | {'SMOOTH (R2)'}")
    print("-" * 65)

    for (symbol, timeframe), group in pairs:
        total_pnl = group['pnl_net'].sum()
        count = len(group)
        wins = len(group[group['pnl_net'] > 0])
        win_rate = (wins / count) * 100
        
        # Check Curve Smoothness
        r_squared = analyze_curve_smoothness(group['pnl_net'])
        
        mode = None
        
        # 1. Check for Winner
        if (total_pnl > MIN_PNL_WIN) and (win_rate > MIN_WIN_RATE) and (count > MIN_TRADES) and (r_squared > 0.8):
            mode = "normal"
            
        # 2. Check for Perfect Loser (Reverse Candidate)
        # We want high confidence in the loss (high R-squared means consistent losing)
        elif (total_pnl < MAX_PNL_LOSS) and (win_rate < MAX_WIN_RATE_LOSS) and (count > MIN_TRADES) and (r_squared > 0.8):
            mode = "reverse"
            
        if mode:
            print(f"{symbol} {timeframe:<5} | {mode:<8} | ${total_pnl:<9.0f} | {win_rate:<6.1f} | {r_squared:.2f}")
            
            if symbol not in whitelist_config:
                whitelist_config[symbol] = {}
            
            # Store config: {"5m": "normal", "1h": "reverse"}
            whitelist_config[symbol][timeframe] = mode

    # Save to JSON
    with open('smart_whitelist.json', 'w') as f:
        json.dump(whitelist_config, f, indent=4)
        
    print(f"\nGenerated 'smart_whitelist.json'.")

if __name__ == "__main__":
    generate_smart_whitelist()
