# trader.py
#
# Description:
# The live execution engine for the algorithmic trading bot. This version
# uses the backward-compatible `tz_aware=True` parameter to ensure the
# MongoDB client correctly handles timezones, preventing crashes.
#
# Author: Gemini
# Date: 2025-09-23 (v13 - Backward-Compatible Timezone Fix)

import ccxt
import pandas as pd
import json
import time
import logging
import os
import importlib
import sys
import uuid
import argparse
import re
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient, errors
from threading import Thread, Event
# BSON CodecOptions is no longer needed with this fix
# from bson.codec_options import CodecOptions

# --- Main Configuration ---
try:
    import config as main_config
except ImportError:
    print("FATAL: Could not find 'config.py'. Please ensure it exists.")
    exit(1)

DATA_POLLING_INTERVAL_S = 10; DATA_POLLING_TIMEOUT_S = 120
TRADE_ENTRY_WINDOW_S = 120; LOCK_HEARTBEAT_INTERVAL_S = 10; LOCK_EXPIRY_S = 30 

# --- Logging & Exception (Unchanged) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trader.log"), logging.StreamHandler()])
logger = logging.getLogger('CryptoTrader')
class LockLostError(Exception): pass

# --- Helper Functions (Unchanged) ---
def load_config(filepath):
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except FileNotFoundError: logger.error(f"Config file not found: {filepath}"); return None
    except json.JSONDecodeError: logger.error(f"Error decoding JSON: {filepath}"); return None

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def load_strategy_class(strategy_name):
    try:
        module_name = to_snake_case(strategy_name)
        module_path = f"strategies.{module_name}"
        strategy_module = importlib.import_module(module_path)
        return getattr(strategy_module, strategy_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading strategy: {strategy_name}. Error: {e}"); return None

# --- MongoLockManager (Unchanged from previous correct version) ---
class MongoLockManager:
    def __init__(self, db_client, trader_id, instance_id):
        self.trader_id = trader_id; self.instance_id = instance_id
        self._stop_heartbeat = Event(); self._heartbeat_thread = None
        self.db = db_client[main_config.MONGO_DB_NAME]
        self.collection = self.db['trader_locks']
        logger.info(f"Lock Manager initialized. Instance ID: {self.instance_id}")

    def acquire(self):
        now = datetime.now(timezone.utc)
        lock_doc = self.collection.find_one({'_id': self.trader_id})
        update_doc = {'$set': {'heartbeat': now, 'instance_id': self.instance_id}}
        if lock_doc:
            last_heartbeat = lock_doc.get('heartbeat')
            if now - last_heartbeat > timedelta(seconds=LOCK_EXPIRY_S):
                logger.warning(f"Found stale lock for {self.trader_id}. Taking over.")
                self.collection.update_one({'_id': self.trader_id}, update_doc)
                self._start_heartbeat()
                return True
            else:
                logger.error(f"Another instance ({lock_doc.get('instance_id')}) of {self.trader_id} is running. Aborting.")
                return False
        else:
            try:
                self.collection.insert_one({'_id': self.trader_id, 'heartbeat': now, 'instance_id': self.instance_id})
                logger.info(f"Lock acquired by instance {self.instance_id}.")
                self._start_heartbeat()
                return True
            except errors.DuplicateKeyError:
                logger.error(f"Another instance just acquired the lock. Aborting.")
                return False
    def verify(self):
        try:
            lock_doc = self.collection.find_one({'_id': self.trader_id})
            return lock_doc and lock_doc.get('instance_id') == self.instance_id
        except Exception: return False 
    def release(self):
        logger.info(f"Instance {self.instance_id} releasing lock for {self.trader_id}...")
        self._stop_heartbeat.set()
        if self._heartbeat_thread: self._heartbeat_thread.join()
        try:
            self.collection.delete_one({'_id': self.trader_id, 'instance_id': self.instance_id})
            logger.info("Lock released successfully.")
        except Exception as e: logger.error(f"Error releasing lock: {e}")
    def _start_heartbeat(self):
        self._stop_heartbeat.clear()
        self._heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Heartbeat thread started.")
    def _heartbeat_loop(self):
        while not self._stop_heartbeat.is_set():
            try:
                result = self.collection.update_one(
                    {'_id': self.trader_id, 'instance_id': self.instance_id},
                    {'$set': {'heartbeat': datetime.now(timezone.utc)}})
                if result.matched_count == 0:
                    logger.critical("Heartbeat failed: Lock taken by another instance! Halting heartbeat.")
                    break
                logger.debug("Heartbeat sent.")
            except Exception as e: logger.error(f"Failed to send heartbeat: {e}")
            time.sleep(LOCK_HEARTBEAT_INTERVAL_S)
        logger.info("Heartbeat thread stopped.")

# --- State Manager (Unchanged) ---
class MongoStateManager:
    def __init__(self, db_client, trader_id):
        self.trader_id = trader_id
        self.db = db_client[main_config.MONGO_DB_NAME]
        self.collection = self.db['trader_states']
        logger.info("State Manager initialized.")
    def save_state(self, state_data):
        try:
            state_data['last_updated'] = datetime.now(timezone.utc)
            self.collection.update_one({'_id': self.trader_id}, {'$set': state_data}, upsert=True)
            logger.debug(f"State saved for trader '{self.trader_id}'.")
        except Exception as e: logger.error(f"Failed to save state: {e}")
    def load_state(self):
        try:
            state = self.collection.find_one({'_id': self.trader_id})
            if state: logger.info(f"Found existing state for '{self.trader_id}'.")
            else: logger.info(f"No state found for '{self.trader_id}'.")
            return state
        except Exception as e: logger.error(f"Failed to load state: {e}"); return None

# --- Main Trader Class (Unchanged) ---
class Trader:
    def __init__(self, config, db_client):
        self.config = config; self.symbol = config['symbol']; self.timeframe = config['timeframe']
        sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', self.symbol)
        self.trader_id = f"{config['strategy_name']}_{sanitized_symbol}_{self.timeframe}"
        logger.info(f"Sanitized Trader ID: {self.trader_id}")
        
        self.instance_id = str(uuid.uuid4())
        self.lock_manager = MongoLockManager(db_client, self.trader_id, self.instance_id)
        self.state_manager = MongoStateManager(db_client, self.trader_id)
        
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'), 'secret': os.getenv('BINANCE_SECRET_KEY'),
            'options': {'defaultType': 'future'},
        })
        self.exchange.set_sandbox_mode(True)
        self.load_market_data()
        StrategyClass = load_strategy_class(config['strategy_name'])
        if not StrategyClass: raise ValueError("Could not load strategy class.")
        self.strategy = StrategyClass(config['parameters'])
        self.load_and_set_initial_state()
        self.timeframe_in_ms = self.exchange.parse_timeframe(self.timeframe) * 1000

    def load_market_data(self):
        try:
            logger.info(f"Loading market data for {self.symbol}...")
            self.exchange.load_markets()
            market = self.exchange.market(self.symbol)
            self.amount_precision = market['precision']['amount']
            self.min_trade_amount = market['limits']['amount']['min']
            logger.info(f"Market data loaded: Min amount={self.min_trade_amount}, Amount precision={self.amount_precision}")
        except Exception as e:
            logger.error(f"Could not load market data for {self.symbol}. Error: {e}"); raise
            
    def load_and_set_initial_state(self):
        saved_state = self.state_manager.load_state()
        if saved_state:
            self.in_position = saved_state.get('in_position', False)
            self.position_type = saved_state.get('position_type')
            self.entry_price = saved_state.get('entry_price', 0)
            self.trade_size_in_asset = saved_state.get('trade_size_in_asset', 0)
            last_ts = saved_state.get('last_candle_timestamp')
            # The database now returns aware datetimes, so no need to localize
            self.last_candle_timestamp = pd.to_datetime(last_ts) if last_ts else None
            logger.info(f"State restored. In Position: {self.in_position}")
        else:
            self.in_position = False; self.position_type = None
            self.entry_price = 0; self.trade_size_in_asset = 0
            self.last_candle_timestamp = None
            logger.info("Initialized with fresh state.")

    def get_current_state_dict(self):
        return {
            'in_position': self.in_position, 'position_type': self.position_type,
            'entry_price': self.entry_price, 'trade_size_in_asset': self.trade_size_in_asset,
            'last_candle_timestamp': self.last_candle_timestamp
        }
    
    def run(self):
        if not self.lock_manager.acquire(): sys.exit(1)
        try:
            logger.info("--- Starting Live Trading Engine ---")
            if self.last_candle_timestamp is None: self.update_latest_data()
            while True:
                self.sleep_until_next_candle()
                self.update_latest_data()
        except LockLostError:
             logger.critical("SHUTTING DOWN: Lock lost. Another instance has taken over.")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt. Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Unexpected critical error: {e}. Shutting down.", exc_info=True)
        finally:
            self.lock_manager.release()
            
    def update_latest_data(self):
        if not self.lock_manager.verify(): raise LockLostError()
        df = self.fetch_latest_candle_data_with_polling()
        if df is not None:
            self.last_candle_timestamp = df.index[-1]
            self.state_manager.save_state(self.get_current_state_dict())
            self.check_for_signals_and_manage_position(df)
        else: logger.warning("Failed to fetch new data after polling. Will try again next cycle.")

    def fetch_latest_candle_data_with_polling(self):
        start_time = time.time(); logger.info("Starting polling cycle...")
        while time.time() - start_time < DATA_POLLING_TIMEOUT_S:
            try:
                candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=250)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                if self.last_candle_timestamp is None or df.index[-1] > self.last_candle_timestamp: return df
                else: time.sleep(DATA_POLLING_INTERVAL_S)
            except Exception as e: logger.warning(f"Error during polling: {e}. Retrying..."); time.sleep(DATA_POLLING_INTERVAL_S)
        logger.error("Data polling timed out."); return None

    def sleep_until_next_candle(self):
        if self.last_candle_timestamp is None: return
        now_utc = datetime.now(timezone.utc)
        next_candle_open_time = self.last_candle_timestamp + timedelta(milliseconds=self.timeframe_in_ms)
        wakeup_time = next_candle_open_time + timedelta(seconds=2)
        sleep_duration = (wakeup_time - now_utc).total_seconds()
        if sleep_duration > 0:
            logger.info(f"Next candle at {next_candle_open_time:%Y-%m-%d %H:%M:%S %Z}. Sleeping for {sleep_duration:.2f}s...")
            time.sleep(sleep_duration)
            logger.info("Waking up.")

    def check_for_signals_and_manage_position(self, df):
        _, df_with_indicators = self.strategy.calculate_indicators(None, df.copy())
        df_with_indicators.dropna(inplace=True)
        if df_with_indicators.empty: return
        prev_row, current_row = df_with_indicators.iloc[-2], df_with_indicators.iloc[-1]
        if self.in_position: self.check_for_exit(current_row['close'])
        else:
            signal_age = (datetime.now(timezone.utc) - current_row.name).total_seconds()
            if signal_age <= TRADE_ENTRY_WINDOW_S: self.check_for_entry(prev_row, current_row)

    def check_for_entry(self, prev_row, current_row):
        if not self.lock_manager.verify(): raise LockLostError()
        signal = self.strategy.get_entry_signal(prev_row, current_row)
        if signal:
            logger.info(f"!!! ENTRY SIGNAL DETECTED: {signal} !!!")
            trade_size_usd = 200; price = current_row['close']
            initial_amount = trade_size_usd / price
            if initial_amount < self.min_trade_amount:
                logger.warning(f"Desired size {initial_amount:.8f} is below minimum {self.min_trade_amount}. Skipping."); return
            self.trade_size_in_asset = self.exchange.amount_to_precision(self.symbol, initial_amount)
            try:
                order = self.exchange.create_market_order(self.symbol, 'buy' if signal == 'LONG' else 'sell', self.trade_size_in_asset)
                self.in_position, self.position_type, self.entry_price = True, signal, order['price']
                logger.info(f"--- TRADE EXECUTED --- New Position: {self.position_type} @ {self.entry_price}")
                self.state_manager.save_state(self.get_current_state_dict())
            except Exception as e: logger.error(f"Error placing order: {e}", exc_info=True); self.in_position = False

    def check_for_exit(self, current_price):
        if not self.lock_manager.verify(): raise LockLostError()
        exit_reason, params = None, self.config['parameters']
        stop_loss_pct, take_profit_pct = params['stop_loss_percent'], stop_loss_pct * params.get('rr_ratio', 2.0)
        if self.position_type == 'LONG':
            if current_price <= self.entry_price * (1 - stop_loss_pct): exit_reason = "Stop-Loss"
            elif current_price >= self.entry_price * (1 + take_profit_pct): exit_reason = "Take-Profit"
        elif self.position_type == 'SHORT':
            if current_price >= self.entry_price * (1 + stop_loss_pct): exit_reason = "Stop-Loss"
            elif current_price <= self.entry_price * (1 - take_profit_pct): exit_reason = "Take-Profit"
        if exit_reason:
            logger.info(f"!!! EXIT SIGNAL DETECTED: {exit_reason} !!!")
            try:
                order = self.exchange.create_market_order(self.symbol, 'sell' if self.position_type == 'LONG' else 'buy', self.trade_size_in_asset)
                pnl = ((order['price'] - self.entry_price) if self.position_type == 'LONG' else (self.entry_price - order['price'])) * self.trade_size_in_asset
                logger.info(f"--- POSITION CLOSED --- PnL: ${pnl:.2f}")
                self.in_position, self.position_type, self.entry_price, self.trade_size_in_asset = False, None, 0, 0
                self.state_manager.save_state(self.get_current_state_dict())
            except Exception as e: logger.error(f"Error closing position: {e}", exc_info=True)


# --- FIX: Main execution block uses tz_aware=True for compatibility ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live crypto trading bot.')
    parser.add_argument('--config', type=str, required=True, help='Path to the strategy JSON config file.')
    args = parser.parse_args()

    config = load_config(args.config)
    db_client = None
    if config:
        try:
            logger.info("Establishing timezone-aware connection to MongoDB...")
            # This is the crucial, backward-compatible change
            db_client = MongoClient(main_config.MONGO_URI, 
                                    serverSelectionTimeoutMS=5000,
                                    tz_aware=True)
            db_client.server_info() # Test connection
            logger.info("MongoDB connection successful.")
            
            trader = Trader(config, db_client)
            trader.run()
        except errors.ConnectionFailure as e:
            logger.critical(f"Could not connect to MongoDB. Shutting down. Error: {e}")
        except Exception as e:
            logger.critical(f"Failed to initialize or run the Trader. Shutting down. Error: {e}", exc_info=True)
        finally:
            if db_client:
                db_client.close()
                logger.info("MongoDB connection closed.")

