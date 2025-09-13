# backtester.py
#
# Description:
# A simple backtesting engine to test a multi-timeframe trading strategy.
# - Higher Timeframe (HTF): Uses ADX to identify the main trend direction.
# - Lower Timeframe (LTF): Uses Ichimoku Cloud signals to pinpoint entries.
#
# Author: Gemini
# Date: 2025-07-26 (v4 - Implemented Ichimoku entry strategy)

import pandas as pd
import pandas_ta as ta
from pymongo import MongoClient
import config  # Import settings from our config.py file

# --- Backtesting Configuration ---
# You can adjust these parameters to test different variations.

# -- Strategy Pair --
# The symbol and timeframes to test. Must exist in your database.
SYMBOL = 'ETH/USDT'
HTF = '1d'  # Higher timeframe for trend direction
LTF = '1h' # Lower timeframe for entry signals

# -- ADX Parameters (HTF) --
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # ADX value above which a strong trend is considered present

# -- Ichimoku Parameters (LTF) --
# Default pandas-ta values are 9, 26, 52 for Tenkan, Kijun, Senkou B
# We will use the standard Ichimoku calculation.

# -- Trade Parameters --
INITIAL_CAPITAL = 10000  # Starting capital for the backtest
TRADE_SIZE_USD = 1000    # Fixed amount in USD for each trade
FEE_PERCENT = 0.001      # Trading fee (e.g., 0.1%)
STOP_LOSS_PERCENT = 0.02 # 2% stop loss
TAKE_PROFIT_PERCENT = 0.04 # 4% take profit

# --- 1. Data Loading ---
# Connects to MongoDB and loads the OHLCV data for our strategy.

def load_data_from_mongo(symbol, timeframe):
    """
    Loads OHLCV data from the MongoDB database.
    """
    print(f"Loading data for {symbol} ({timeframe}) from MongoDB...")
    try:
        client = MongoClient(config.MONGO_URI)
        db = client[config.MONGO_DB_NAME]
        collection_name = f"{symbol.replace('/', '_')}_{timeframe}"
        
        if collection_name not in db.list_collection_names():
            print(f"Error: Collection '{collection_name}' not found in the database.")
            return pd.DataFrame()

        # Load all data from the collection and sort by timestamp
        data = list(db[collection_name].find({}, {'_id': 0}).sort('timestamp', 1))
        
        if not data:
            print(f"No data found in collection '{collection_name}'.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        print(f"Successfully loaded {len(df)} records.")
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

# --- 2. Indicator Calculation ---

def calculate_indicators(df_htf, df_ltf):
    """
    Calculates the required technical indicators for both timeframes.
    """
    print("Calculating technical indicators...")
    # Calculate ADX on the Higher Timeframe (HTF)
    df_htf.ta.adx(length=ADX_PERIOD, append=True)
    
    # Calculate Ichimoku Cloud on the Lower Timeframe (LTF)
    # This will append multiple columns (ISA_9, ISB_26, ITS_9, IKS_26, ICS_26)
    df_ltf.ta.ichimoku(append=True)
    
    # Clean up any rows with missing indicator values
    df_htf.dropna(inplace=True)
    df_ltf.dropna(inplace=True)
    print("Indicators calculated.")
    return df_htf, df_ltf

# --- 3. Backtesting Logic ---

def run_backtest(df_htf, df_ltf):
    """
    Runs the backtest simulation based on the multi-timeframe strategy.
    """
    print("\n--- Starting Backtest ---")
    capital = INITIAL_CAPITAL
    trades = []
    position = None  # Can be 'LONG', 'SHORT', or None

    # --- Convert CCXT timeframe to modern Pandas frequency string ---
    pandas_ltf_freq = LTF.replace('m', 'min')

    # Resample HTF data to match LTF index for easy lookup
    print(f"Resampling HTF data to {pandas_ltf_freq} frequency...")
    htf_trend = df_htf[[f'ADX_{ADX_PERIOD}', f'DMP_{ADX_PERIOD}', f'DMN_{ADX_PERIOD}']].resample(pandas_ltf_freq).ffill()
    
    # Align the LTF data with the resampled HTF trend data
    aligned_df = df_ltf.join(htf_trend).dropna()
    print(f"Data aligned. Backtest will run on {len(aligned_df)} candles.")

    if aligned_df.empty:
        print("Warning: Aligned DataFrame is empty. No trades can be executed. Check data and timeframes.")
        return capital, trades
    
    # Ichimoku column names from pandas_ta
    tenkan_col = 'ITS_9'
    kijun_col = 'IKS_26'
    senkou_a_col = 'ISA_9'
    senkou_b_col = 'ISB_26'

    for i in range(1, len(aligned_df)):
        prev_row = aligned_df.iloc[i-1]
        current_row = aligned_df.iloc[i]

        # --- Check Exit Conditions First ---
        if position:
            pnl = 0
            exit_reason = None
            if position == 'LONG':
                if current_row['low'] <= entry_price * (1 - STOP_LOSS_PERCENT):
                    pnl = (entry_price * (1 - STOP_LOSS_PERCENT)) - entry_price
                    exit_reason = "Stop Loss"
                elif current_row['high'] >= entry_price * (1 + TAKE_PROFIT_PERCENT):
                    pnl = (entry_price * (1 + TAKE_PROFIT_PERCENT)) - entry_price
                    exit_reason = "Take Profit"
            elif position == 'SHORT':
                if current_row['high'] >= entry_price * (1 + STOP_LOSS_PERCENT):
                    pnl = entry_price - (entry_price * (1 + STOP_LOSS_PERCENT))
                    exit_reason = "Stop Loss"
                elif current_row['low'] <= entry_price * (1 - TAKE_PROFIT_PERCENT):
                    pnl = entry_price - (entry_price * (1 - TAKE_PROFIT_PERCENT))
                    exit_reason = "Take Profit"
            
            if exit_reason:
                trade_pnl_usd = (pnl / entry_price) * TRADE_SIZE_USD
                fee = TRADE_SIZE_USD * FEE_PERCENT * 2 # Entry and Exit fee
                net_pnl = trade_pnl_usd - fee
                capital += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_row.name,
                    'pnl': net_pnl,
                    'reason': exit_reason
                })
                position = None

        # --- Check Entry Conditions ---
        if not position:
            # Trend Condition (from HTF)
            is_trending = current_row[f'ADX_{ADX_PERIOD}'] > ADX_THRESHOLD
            is_uptrend = current_row[f'DMP_{ADX_PERIOD}'] > current_row[f'DMN_{ADX_PERIOD}']
            
            # Ichimoku Entry Conditions (from LTF)
            # Bullish: Price above cloud AND Tenkan crosses above Kijun
            price_above_cloud = current_row['close'] > current_row[senkou_a_col] and \
                                current_row['close'] > current_row[senkou_b_col]
            
            bullish_tk_cross = prev_row[tenkan_col] < prev_row[kijun_col] and \
                               current_row[tenkan_col] > current_row[kijun_col]

            # Bearish: Price below cloud AND Tenkan crosses below Kijun
            price_below_cloud = current_row['close'] < current_row[senkou_a_col] and \
                                current_row['close'] < current_row[senkou_b_col]

            bearish_tk_cross = prev_row[tenkan_col] > prev_row[kijun_col] and \
                               current_row[tenkan_col] < current_row[kijun_col]

            if is_trending and is_uptrend and price_above_cloud and bullish_tk_cross:
                position = 'LONG'
                entry_price = current_row['close']
                entry_date = current_row.name
                print(f"{entry_date}: Enter LONG at {entry_price:.2f}")

            elif is_trending and not is_uptrend and price_below_cloud and bearish_tk_cross:
                position = 'SHORT'
                entry_price = current_row['close']
                entry_date = current_row.name
                print(f"{entry_date}: Enter SHORT at {entry_price:.2f}")

    return capital, trades

# --- 4. Reporting ---

def print_results(final_capital, trades):
    """
    Prints a summary of the backtest performance.
    """
    print("\n--- Backtest Results ---")
    pnl = final_capital - INITIAL_CAPITAL
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Gross P&L:       ${pnl:,.2f}")
    
    if trades:
        total_trades = len(trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        print(f"Total Trades:    {total_trades}")
        print(f"Win Rate:        {win_rate:.2f}%")
        if wins:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            print(f"Average Win:     ${avg_win:,.2f}")
        if losses:
            avg_loss = sum(t['pnl'] for t in losses) / len(losses)
            print(f"Average Loss:    ${avg_loss:,.2f}")
    else:
        print("No trades were executed.")
    print("------------------------")


# --- Main Execution ---

if __name__ == "__main__":
    # To run this script:
    # 1. Make sure you have run the data ingestion engine first.
    # 2. Install required library:
    #    pip install pandas_ta
    # 3. (Optional) Adjust parameters in the "Backtesting Configuration" section.
    # 4. Run the script from your activated venv:
    #    python backtester.py
    
    # 1. Load data for both timeframes
    df_htf = load_data_from_mongo(SYMBOL, HTF)
    df_ltf = load_data_from_mongo(SYMBOL, LTF)

    if df_htf.empty or df_ltf.empty:
        print("Could not load data for backtest. Exiting.")
    else:
        # 2. Calculate indicators
        df_htf, df_ltf = calculate_indicators(df_htf, df_ltf)
        
        # 3. Run the backtest
        final_capital, trades = run_backtest(df_htf, df_ltf)
        
        # 4. Print the results
        print_results(final_capital, trades)
