# control_panel.py
#
# Description:
# A flexible command-line tool to remotely control live trading bots. This
# version corrects the logic to ensure `trade_size` settings are applied
# to the correct trader documents.
#
# Author: Gemini
# Date: 2025-10-04

import argparse
import re
from pymongo import MongoClient, errors

# --- Main Configuration ---
try:
    import config as main_config
except ImportError:
    print("FATAL: Could not find 'config.py'.")
    exit(1)

def control_traders(filter_criteria, mode=None, trade_size=None):
    db_client = None
    try:
        print("Connecting to MongoDB...")
        db_client = MongoClient(main_config.MONGO_URI, serverSelectionTimeoutMS=5000)
        db = db_client[main_config.MONGO_DB_NAME]
        locks_collection = db['trader_locks']
        
        print("Searching for active traders matching criteria...")
        active_traders = list(locks_collection.find(filter_criteria, {'_id': 1}))

        if not active_traders:
            print("No active traders found matching the specified criteria."); return

        target_ids = [trader['_id'] for trader in active_traders]
        print(f"Found {len(target_ids)} active trader(s) to update:")
        for trader_id in target_ids: print(f"  - {trader_id}")

        if mode:
            print(f"\nSetting operational mode to '{mode}' for all targets...")
            controls_collection = db['trader_controls']
            for trader_id in target_ids:
                controls_collection.update_one({'_id': trader_id}, {'$set': {'operational_mode': mode}}, upsert=True)
            print("Operational mode command sent successfully.")

        if trade_size is not None:
            print(f"\nSetting trade size to ${trade_size} for all targets...")
            settings_collection = db['trader_settings']
            # --- FIX: Use the robust update_one loop for trade size as well ---
            for trader_id in target_ids:
                settings_collection.update_one({'_id': trader_id}, {'$set': {'trade_size_usd': trade_size}}, upsert=True)
            print("Trade size command sent successfully.")

    except errors.ConnectionFailure as e:
        print(f"CRITICAL: Could not connect to MongoDB. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if db_client: db_client.close(); print("MongoDB connection closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remotely control a live trading bot or a group of bots.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--exchange', required=True, type=str, help='The target exchange (e.g., binance, bybit).')
    parser.add_argument('--strategy', required=False, type=str, help='The name of the strategy to target.')
    parser.add_argument('--symbol', required=False, type=str, help='The trading symbol to target.')
    parser.add_argument('--timeframe', required=False, type=str, help='The timeframe to target.')
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('--mode', type=str, choices=['trading', 'standby', 'exit_all'], help='The operational mode to set.')
    command_group.add_argument('--trade-size', type=float, help='The trade size in USD (e.g., 250.50).')
    args = parser.parse_args()
    
    filter_parts = [{'_id': {'$regex': f"^{args.exchange.lower()}_"}}]
    if args.strategy: filter_parts.append({'_id': {'$regex': f"_{args.strategy}_"}})
    if args.symbol:
        sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', args.symbol)
        filter_parts.append({'_id': {'$regex': f"_{sanitized_symbol}_"}})
    if args.timeframe: filter_parts.append({'_id': {'$regex': f"_{args.timeframe}$"}})
    query_filter = {'$and': filter_parts} if len(filter_parts) > 1 else filter_parts[0]
    
    print("Constructed MongoDB Query Filter:", query_filter)
    control_traders(query_filter, args.mode, args.trade_size)

