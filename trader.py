# trader.py
#
# Description:
# The complete, production-ready live execution engine for the algorithmic
# trading platform. This script is the final version, incorporating all
# features including command-line configuration, a distributed lock with
# fencing tokens, persistent state management, a single shared database
# client, an intelligent run loop, dynamic market rules, smart trade sizing,
# correct data type handling, remote control capabilities, and a permanent
# trade history log.
#
# Author: Gemini
# Date: 2025-09-24 (v16 - Final Production Version)

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

# --- Main Configuration ---
try:
    import config as main_config
except ImportError:
    # Use a basic logger for this critical startup error
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical("FATAL: Could not find 'config.py'. Please ensure it exists.")
    exit(1)

DATA_POLLING_INTERVAL_S = 10
DATA_POLLING_TIMEOUT_S = 120
TRADE_ENTRY_WINDOW_S = 120
LOCK_HEARTBEAT_INTERVAL_S = 10
LOCK_EXPIRY_S = 30

# --- Contextual Logging Setup ---
def setup_logging(trader_id):
    """Configures the logger to include the trader_id in every message."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(trader_id)s] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trader.log"),
            logging.StreamHandler()
        ]
    )
    return logging.LoggerAdapter(logging.getLogger(__name__), {'trader_id': trader_id})

# --- Custom Exception ---
class LockLostError(Exception):
    """Custom exception to signal that lock ownership has been lost."""
    pass

# --- Helper Functions ---
def load_config(filepath):
    """Loads the strategy JSON configuration file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Use a basic logger as the contextual one may not exist yet
        logging.error(f"Configuration file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {filepath}")
        return None

def to_snake_case(name):
    """Converts CamelCase to snake_case for dynamic module loading."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def load_strategy_class(strategy_name):
    """Dynamically imports and returns a strategy class by name."""
    try:
        module_name = to_snake_case(strategy_name)
        module_path = f"strategies.{module_name}"
        strategy_module = importlib.import_module(module_path)
        return getattr(strategy_module, strategy_name)
    except (ImportError, AttributeError):
        logging.error(f"Error loading strategy '{strategy_name}' from module '{module_path}'.")
        return None

# --- Distributed Lock Manager ---
class MongoLockManager:
    """Handles acquiring and maintaining a distributed lock in MongoDB."""
    def __init__(self, db_client, trader_id, instance_id, logger):
        self.trader_id = trader_id
        self.instance_id = instance_id  # Fencing token
        self.logger = logger
        self._stop_heartbeat = Event()
        self._heartbeat_thread = None
        self.db = db_client[main_config.MONGO_DB_NAME]
        self.collection = self.db['trader_locks']
        self.logger.info(f"Lock Manager initialized. Instance ID: {self.instance_id}")

    def acquire(self):
        now = datetime.now(timezone.utc)
        lock_doc = self.collection.find_one({'_id': self.trader_id})
        update_doc = {'$set': {'heartbeat': now, 'instance_id': self.instance_id}}
        if lock_doc:
            last_heartbeat = lock_doc.get('heartbeat')
            if now - last_heartbeat > timedelta(seconds=LOCK_EXPIRY_S):
                self.logger.warning(f"Found stale lock for {self.trader_id}. Taking over.")
                self.collection.update_one({'_id': self.trader_id}, update_doc)
                self._start_heartbeat()
                return True
            else:
                self.logger.error(f"Another instance ({lock_doc.get('instance_id')}) of {self.trader_id} is running. Aborting.")
                return False
        else:
            try:
                self.collection.insert_one({'_id': self.trader_id, 'heartbeat': now, 'instance_id': self.instance_id})
                self.logger.info(f"Lock acquired by instance {self.instance_id}.")
                self._start_heartbeat()
                return True
            except errors.DuplicateKeyError:
                self.logger.error(f"Another instance just acquired the lock. Aborting.")
                return False

    def verify(self):
        """Verifies that this instance still holds the lock."""
        try:
            lock_doc = self.collection.find_one({'_id': self.trader_id})
            return lock_doc and lock_doc.get('instance_id') == self.instance_id
        except Exception:
            return False

    def release(self):
        """Releases the lock and stops the heartbeat."""
        self.logger.info(f"Instance {self.instance_id} releasing lock...")
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join()
        try:
            self.collection.delete_one({'_id': self.trader_id, 'instance_id': self.instance_id})
            self.logger.info("Lock released successfully.")
        except Exception as e:
            self.logger.error(f"Error releasing lock: {e}")

    def _start_heartbeat(self):
        self._stop_heartbeat.clear()
        self._heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self.logger.info("Heartbeat thread started.")

    def _heartbeat_loop(self):
        while not self._stop_heartbeat.is_set():
            try:
                result = self.collection.update_one(
                    {'_id': self.trader_id, 'instance_id': self.instance_id},
                    {'$set': {'heartbeat': datetime.now(timezone.utc)}})
                if result.matched_count == 0:
                    self.logger.critical("Heartbeat failed: Lock taken by another instance! Halting heartbeat.")
                    break
                self.logger.debug("Heartbeat sent.")
            except Exception as e:
                self.logger.error(f"Failed to send heartbeat: {e}")
            time.sleep(LOCK_HEARTBEAT_INTERVAL_S)
        self.logger.info("Heartbeat thread stopped.")

# --- Persistent State Manager ---
class MongoStateManager:
    """Handles saving state and trade history to MongoDB."""
    def __init__(self, db_client, trader_id, logger):
        self.trader_id = trader_id
        self.logger = logger
        self.db = db_client[main_config.MONGO_DB_NAME]
        self.state_collection = self.db['trader_states']
        self.history_collection = self.db['trade_history']
        self.logger.info("State Manager initialized.")

    def save_state(self, state_data):
        try:
            state_data['last_updated'] = datetime.now(timezone.utc)
            self.state_collection.update_one({'_id': self.trader_id}, {'$set': state_data}, upsert=True)
            self.logger.debug("State saved.")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load_state(self):
        try:
            state = self.state_collection.find_one({'_id': self.trader_id})
            if state: self.logger.info("Found existing state.")
            else: self.logger.info("No state found.")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return None

    def save_trade_history(self, trade_data):
        """Saves a completed trade record to the history collection."""
        try:
            self.history_collection.insert_one(trade_data)
            self.logger.info("Trade history record saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")

# --- Main Trading Class ---
class Trader:
    def __init__(self, config, db_client, logger):
        self.config = config
        self.symbol = config['symbol']
        self.timeframe = config['timeframe']
        self.logger = logger
        
        sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', self.symbol)
        self.trader_id = f"{config['strategy_name']}_{sanitized_symbol}_{self.timeframe}"
        self.logger.info(f"Initializing trader with ID: {self.trader_id}")
        
        self.instance_id = str(uuid.uuid4())
        self.lock_manager = MongoLockManager(db_client, self.trader_id, self.instance_id, self.logger)
        self.state_manager = MongoStateManager(db_client, self.trader_id, self.logger)
        
        self.control_collection = db_client[main_config.MONGO_DB_NAME]['trader_controls']
        self.operational_mode = 'trading'
        
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'options': {'defaultType': 'future'},
        })
        self.exchange.set_sandbox_mode(True)
        self.load_market_data()

        StrategyClass = load_strategy_class(strategy_name=config['strategy_name'])
        if not StrategyClass:
            raise ValueError("Could not load strategy class.")
        self.strategy = StrategyClass(config['parameters'])
        
        self.load_and_set_initial_state()
        self.timeframe_in_ms = self.exchange.parse_timeframe(self.timeframe) * 1000

    def load_market_data(self):
        try:
            self.logger.info(f"Loading market data for {self.symbol}...")
            self.exchange.load_markets()
            market = self.exchange.market(self.symbol)
            self.amount_precision = market['precision']['amount']
            self.min_trade_amount = market['limits']['amount']['min']
            self.logger.info(f"Market data loaded: Min amount={self.min_trade_amount}, Amount precision={self.amount_precision}")
        except Exception as e:
            self.logger.error(f"Could not load market data for {self.symbol}. Error: {e}")
            raise
            
    def load_and_set_initial_state(self):
        saved_state = self.state_manager.load_state()
        if saved_state:
            self.in_position = saved_state.get('in_position', False)
            self.position_type = saved_state.get('position_type')
            self.entry_price = saved_state.get('entry_price', 0)
            self.trade_size_in_asset = saved_state.get('trade_size_in_asset', 0)
            self.entry_time = saved_state.get('entry_time')
            last_ts = saved_state.get('last_candle_timestamp')
            self.last_candle_timestamp = pd.to_datetime(last_ts) if last_ts else None
            self.logger.info(f"State restored. In Position: {self.in_position}")
        else:
            self.in_position = False
            self.position_type = None
            self.entry_price = 0
            self.trade_size_in_asset = 0
            self.last_candle_timestamp = None
            self.entry_time = None
            self.logger.info("Initialized with fresh state.")

    def get_current_state_dict(self):
        return {
            'in_position': self.in_position,
            'position_type': self.position_type,
            'entry_price': self.entry_price,
            'trade_size_in_asset': self.trade_size_in_asset,
            'last_candle_timestamp': self.last_candle_timestamp,
            'entry_time': self.entry_time
        }
    
    def check_control_signals(self):
        """Checks MongoDB for any commands from the control panel."""
        try:
            control_doc = self.control_collection.find_one({'_id': self.trader_id})
            if control_doc:
                new_mode = control_doc.get('operational_mode', 'trading')
                if new_mode != self.operational_mode:
                    self.logger.info(f"CONTROL SIGNAL: Changing mode from '{self.operational_mode}' to '{new_mode}'.")
                    self.operational_mode = new_mode
                
                if self.operational_mode == 'exit_all' and self.in_position:
                    self.logger.info("Exit command received. Forcing position closure...")
                    self.check_for_exit(0, force_exit=True)
                    self.operational_mode = 'standby'
                    self.control_collection.update_one({'_id': self.trader_id}, {'$set': {'operational_mode': 'standby'}})
        except Exception as e:
            self.logger.error(f"Could not check control signals: {e}")

    def run(self):
        if not self.lock_manager.acquire():
            sys.exit(1)
        try:
            self.logger.info("--- Starting Live Trading Engine ---")
            if self.last_candle_timestamp is None:
                self.update_latest_data()
            while True:
                self.sleep_until_next_candle()
                self.update_latest_data()
        except LockLostError:
             self.logger.critical("SHUTTING DOWN: Lock lost. Another instance has taken over.")
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt. Shutting down gracefully...")
        except Exception as e:
            self.logger.error(f"Unexpected critical error: {e}. Shutting down.", exc_info=True)
        finally:
            self.lock_manager.release()
            
    def update_latest_data(self):
        if not self.lock_manager.verify():
            raise LockLostError()
        
        self.check_control_signals()
        
        df = self.fetch_latest_candle_data_with_polling()
        if df is not None:
            self.last_candle_timestamp = df.index[-1]
            self.state_manager.save_state(self.get_current_state_dict())
            self.check_for_signals_and_manage_position(df)
        else:
            self.logger.warning("Failed to fetch new data after polling. Will try again next cycle.")

    def fetch_latest_candle_data_with_polling(self):
        start_time = time.time()
        self.logger.info("Starting polling cycle...")
        while time.time() - start_time < DATA_POLLING_TIMEOUT_S:
            try:
                candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=250)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                if self.last_candle_timestamp is None or df.index[-1] > self.last_candle_timestamp:
                    return df
                else:
                    time.sleep(DATA_POLLING_INTERVAL_S)
            except Exception as e:
                self.logger.warning(f"Error during polling: {e}. Retrying...")
                time.sleep(DATA_POLLING_INTERVAL_S)
        self.logger.error("Data polling timed out.")
        return None

    def sleep_until_next_candle(self):
        if self.last_candle_timestamp is None:
            return
        now_utc = datetime.now(timezone.utc)
        next_candle_open_time = self.last_candle_timestamp + timedelta(milliseconds=self.timeframe_in_ms)
        wakeup_time = next_candle_open_time + timedelta(seconds=2)
        sleep_duration = (wakeup_time - now_utc).total_seconds()
        if sleep_duration > 0:
            self.logger.info(f"Next candle at {next_candle_open_time:%Y-%m-%d %H:%M:%S %Z}. Sleeping for {sleep_duration:.2f}s...")
            time.sleep(sleep_duration)
            self.logger.info("Waking up.")

    def check_for_signals_and_manage_position(self, df):
        _, df_with_indicators = self.strategy.calculate_indicators(None, df.copy())
        df_with_indicators.dropna(inplace=True)
        if df_with_indicators.empty:
            return
        prev_row, current_row = df_with_indicators.iloc[-2], df_with_indicators.iloc[-1]
        if self.in_position:
            self.check_for_exit(current_row['close'])
        else:
            signal_age = (datetime.now(timezone.utc) - current_row.name).total_seconds()
            if signal_age <= TRADE_ENTRY_WINDOW_S:
                self.check_for_entry(prev_row, current_row)

    def check_for_entry(self, prev_row, current_row):
        if not self.lock_manager.verify():
            raise LockLostError()
        
        if self.operational_mode != 'trading':
            self.logger.info(f"Mode is '{self.operational_mode}', skipping new entry check.")
            return
            
        signal = self.strategy.get_entry_signal(prev_row, current_row)
        if signal:
            self.logger.info(f"!!! ENTRY SIGNAL DETECTED: {signal} !!!")
            trade_size_usd = 200
            price = current_row['close']
            initial_amount = trade_size_usd / price
            
            final_amount = max(initial_amount, self.min_trade_amount)
            if final_amount > initial_amount:
                self.logger.info(f"Initial size {initial_amount:.6f} was below minimum. Adjusting to {self.min_trade_amount:.6f}.")
            
            self.trade_size_in_asset = final_amount
            formatted_amount = self.exchange.amount_to_precision(self.symbol, self.trade_size_in_asset)
            self.logger.info(f"Calculated final trade size: {formatted_amount}")
            
            try:
                order = self.exchange.create_market_order(self.symbol, 'buy' if signal == 'LONG' else 'sell', formatted_amount)
                self.in_position, self.position_type, self.entry_price = True, signal, order['price']
                self.entry_time = datetime.now(timezone.utc)
                self.logger.info(f"--- TRADE EXECUTED --- New Position: {self.position_type} @ {self.entry_price}")
                self.state_manager.save_state(self.get_current_state_dict())
            except Exception as e:
                self.logger.error(f"Error placing order: {e}", exc_info=True)
                self.in_position = False

    def check_for_exit(self, current_price, force_exit=False):
        if not self.lock_manager.verify():
            raise LockLostError()
            
        exit_reason = None
        params = self.config['parameters']
        
        if force_exit:
            exit_reason = "Manual Exit (via Control Panel)"
        else:
            stop_loss_pct = params['stop_loss_percent']
            take_profit_pct = stop_loss_pct * params.get('rr_ratio', 2.0)
            if self.position_type == 'LONG':
                if current_price <= self.entry_price * (1 - stop_loss_pct):
                    exit_reason = "Stop-Loss"
                elif current_price >= self.entry_price * (1 + take_profit_pct):
                    exit_reason = "Take-Profit"
            elif self.position_type == 'SHORT':
                if current_price >= self.entry_price * (1 + stop_loss_pct):
                    exit_reason = "Stop-Loss"
                elif current_price <= self.entry_price * (1 - take_profit_pct):
                    exit_reason = "Take-Profit"
            
        if exit_reason:
            self.logger.info(f"!!! EXIT SIGNAL DETECTED: {exit_reason} !!!")
            try:
                formatted_amount = self.exchange.amount_to_precision(self.symbol, self.trade_size_in_asset)
                order = self.exchange.create_market_order(self.symbol, 'sell' if self.position_type == 'LONG' else 'buy', formatted_amount)
                
                exit_price = order['price']
                exit_time = datetime.now(timezone.utc)
                pnl = ((exit_price - self.entry_price) if self.position_type == 'LONG' else (self.entry_price - exit_price)) * self.trade_size_in_asset
                self.logger.info(f"--- POSITION CLOSED --- PnL: ${pnl:.2f}")

                trade_record = {
                    'trader_id': self.trader_id,
                    'instance_id': self.instance_id,
                    'symbol': self.symbol,
                    'position_type': self.position_type,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'entry_time': self.entry_time,
                    'exit_time': exit_time,
                    'trade_size': self.trade_size_in_asset,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                }
                self.state_manager.save_trade_history(trade_record)

                self.in_position, self.position_type, self.entry_price, self.trade_size_in_asset, self.entry_time = False, None, 0, 0, None
                self.state_manager.save_state(self.get_current_state_dict())
            except Exception as e:
                self.logger.error(f"Error closing position: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live crypto trading bot.')
    parser.add_argument('--config', type=str, required=True, help='Path to the strategy JSON config file.')
    args = parser.parse_args()

    config = load_config(args.config)
    db_client = None
    logger = logging.getLogger(__name__)

    if config:
        try:
            sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', config['symbol'])
            trader_id = f"{config['strategy_name']}_{sanitized_symbol}_{config['timeframe']}"
            logger = setup_logging(trader_id)
            
            logger.info("Establishing timezone-aware connection to MongoDB...")
            db_client = MongoClient(main_config.MONGO_URI, serverSelectionTimeoutMS=5000, tz_aware=True)
            db_client.server_info()
            logger.info("MongoDB connection successful.")
            
            trader = Trader(config, db_client, logger)
            trader.run()
        except errors.ConnectionFailure as e:
            logger.critical(f"Could not connect to MongoDB. Shutting down. Error: {e}")
        except Exception as e:
            logger.critical(f"Failed to initialize or run the Trader. Error: {e}", exc_info=True)
        finally:
            if db_client:
                db_client.close()
                logger.info("MongoDB connection closed.")

