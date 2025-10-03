# trader.py
#
# Description:
# The complete, final, production-ready, multi-exchange live execution
# engine. This script can run any configured strategy on any CCXT-supported
# exchange.
#
# Author: Gemini
# Date: 2025-10-01 (Final Multi-Exchange Version)

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

from trader_core import MongoLockManager, MongoStateManager, ControlSignalChecker, LockLostError

# --- Main Configuration ---
try:
    import config as main_config
except ImportError:
    logging.basicConfig(level=logging.CRITICAL); logging.critical("FATAL: Could not find 'config.py'.")
    exit(1)

DATA_POLLING_INTERVAL_S = 10; DATA_POLLING_TIMEOUT_S = 120; TRADE_ENTRY_WINDOW_S = 120
DEFAULT_TRADE_SIZE_USD = 20.0 # A smaller, safer default for live trading
MAX_SLIPPAGE_TICKS = 5

def setup_logging(trader_id):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(trader_id)s] - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("trader.log"), logging.StreamHandler()])
    return logging.LoggerAdapter(logging.getLogger(__name__), {'trader_id': trader_id})

def load_config(filepath):
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return None

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name); return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def load_strategy_class(strategy_name):
    try:
        module_name = to_snake_case(strategy_name)
        module_path = f"strategies.{module_name}"
        strategy_module = importlib.import_module(module_path)
        return getattr(strategy_module, strategy_name)
    except (ImportError, AttributeError): return None

class Trader:
    def __init__(self, config, db_client, logger):
        self.config = config; self.symbol = config['symbol']; self.timeframe = config['timeframe']
        self.exchange_id = config['exchange'].lower()
        self.logger = logger
        
        sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', self.symbol)
        self.trader_id = f"{self.exchange_id}_{config['strategy_name']}_{sanitized_symbol}_{self.timeframe}"
        self.logger.info(f"Initializing trader with ID: {self.trader_id}")
        
        self.instance_id = str(uuid.uuid4())
        self.lock_manager = MongoLockManager(db_client, self.trader_id, self.instance_id, self.logger)
        self.state_manager = MongoStateManager(db_client, self.trader_id, self.logger)
        
        self.settings_collection = db_client[main_config.MONGO_DB_NAME]['trader_settings']
        self.trade_size_usd = DEFAULT_TRADE_SIZE_USD
        
        self.operational_mode = 'trading'
        self.control_checker = ControlSignalChecker(db_client, self.trader_id, self, self.logger)
        
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': os.getenv(f"{self.exchange_id.upper()}_API_KEY"),
                'secret': os.getenv(f"{self.exchange_id.upper()}_SECRET_KEY"),
                'options': {'defaultType': 'future'},
            })
            # IMPORTANT: For live trading, REMOVE OR COMMENT OUT the next line.
            # if self.exchange_id == 'binance': self.exchange.set_sandbox_mode(True)
        except AttributeError:
            self.logger.critical(f"Exchange '{self.exchange_id}' is not supported by CCXT.")
            raise
        
        self.load_market_data()
        StrategyClass = load_strategy_class(strategy_name=config['strategy_name'])
        if not StrategyClass: raise ValueError("Could not load strategy class.")
        self.strategy = StrategyClass(config['parameters'])
        self.load_and_set_initial_state()
        self.timeframe_in_ms = self.exchange.parse_timeframe(self.timeframe) * 1000

    def load_market_data(self):
        try:
            self.logger.info(f"Loading market data for {self.symbol} on {self.exchange_id}...")
            self.exchange.load_markets()
            market = self.exchange.market(self.symbol)
            self.amount_precision = market['precision']['amount']
            self.min_trade_amount = market['limits']['amount']['min']
            self.price_precision = market['precision']['price']
            self.tick_size = 10 ** -self.price_precision
            self.logger.info(f"Market data loaded: Min amount={self.min_trade_amount}, Tick Size={self.tick_size}")
        except Exception as e:
            self.logger.error(f"Could not load market data: {e}"); raise
            
    def load_and_set_initial_state(self):
        # ... (This method's logic is unchanged) ...
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
            self.in_position=False; self.position_type=None; self.entry_price=0; 
            self.trade_size_in_asset=0; self.last_candle_timestamp=None; self.entry_time=None
            self.logger.info("Initialized with fresh state.")

    def get_current_state_dict(self):
        # ... (This method's logic is unchanged) ...
        return {
            'in_position': self.in_position, 'position_type': self.position_type,
            'entry_price': self.entry_price, 'trade_size_in_asset': self.trade_size_in_asset,
            'last_candle_timestamp': self.last_candle_timestamp, 'entry_time': self.entry_time
        }
    
    def load_dynamic_settings(self):
        # ... (This method's logic is unchanged) ...
        try:
            settings_doc = self.settings_collection.find_one({'_id': self.trader_id})
            if settings_doc and 'trade_size_usd' in settings_doc:
                new_size = float(settings_doc['trade_size_usd'])
                if new_size != self.trade_size_usd:
                    self.logger.info(f"SETTINGS UPDATE: Trade size changed from ${self.trade_size_usd} to ${new_size}.")
                    self.trade_size_usd = new_size
            else:
                if self.trade_size_usd != DEFAULT_TRADE_SIZE_USD:
                    self.logger.info(f"SETTINGS UPDATE: Reverting to default ${DEFAULT_TRADE_SIZE_USD}.")
                    self.trade_size_usd = DEFAULT_TRADE_SIZE_USD
        except Exception as e: self.logger.error(f"Could not load dynamic settings: {e}")

    def run(self):
        if not self.lock_manager.acquire(): sys.exit(1)
        try:
            self.logger.info("--- Starting Live Trading Engine ---")
            self.control_checker.start()
            if self.last_candle_timestamp is None: self.update_latest_data()
            while True:
                self.sleep_until_next_candle()
                self.update_latest_data()
        except LockLostError: self.logger.critical("SHUTTING DOWN: Lock lost.")
        except KeyboardInterrupt: self.logger.info("Keyboard interrupt. Shutting down...")
        except Exception as e: self.logger.error(f"Unexpected critical error: {e}. Shutting down.", exc_info=True)
        finally:
            self.control_checker.stop(); self.lock_manager.release()
            
    def update_latest_data(self):
        if not self.lock_manager.verify(): raise LockLostError()
        self.load_dynamic_settings()
        df = self.fetch_latest_candle_data_with_polling()
        if df is not None:
            self.last_candle_timestamp = df.index[-1]
            self.state_manager.save_state(self.get_current_state_dict())
            self.check_for_signals_and_manage_position(df)
        else: self.logger.warning("Failed to fetch new data after polling.")

    def fetch_latest_candle_data_with_polling(self):
        # ... (This method's logic is unchanged) ...
        start_time = time.time(); self.logger.info("Starting polling cycle...")
        while time.time() - start_time < DATA_POLLING_TIMEOUT_S:
            try:
                candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=250)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                if self.last_candle_timestamp is None or df.index[-1] > self.last_candle_timestamp: return df
                else: time.sleep(DATA_POLLING_INTERVAL_S)
            except Exception as e: self.logger.warning(f"Error during polling: {e}. Retrying..."); time.sleep(DATA_POLLING_INTERVAL_S)
        self.logger.error("Data polling timed out."); return None

    def sleep_until_next_candle(self):
        # ... (This method's logic is unchanged) ...
        if self.last_candle_timestamp is None: return
        now_utc = datetime.now(timezone.utc)
        next_candle_open_time = self.last_candle_timestamp + timedelta(milliseconds=self.timeframe_in_ms)
        wakeup_time = next_candle_open_time + timedelta(seconds=2)
        sleep_duration = (wakeup_time - now_utc).total_seconds()
        if sleep_duration > 0:
            self.logger.info(f"Next candle at {next_candle_open_time:%Y-%m-%d %H:%M:%S %Z}. Sleeping for {sleep_duration:.2f}s...")
            time.sleep(sleep_duration); self.logger.info("Waking up.")

    def check_for_signals_and_manage_position(self, df):
        # ... (This method's logic is unchanged) ...
        _, df_with_indicators = self.strategy.calculate_indicators(None, df.copy())
        df_with_indicators.dropna(inplace=True);
        if df_with_indicators.empty: return
        prev_row, current_row = df_with_indicators.iloc[-2], df_with_indicators.iloc[-1]
        if self.in_position: self.check_for_exit(current_row['close'])
        if not self.in_position:
            signal_age = (datetime.now(timezone.utc) - current_row.name).total_seconds()
            if signal_age <= TRADE_ENTRY_WINDOW_S: self.check_for_entry(prev_row, current_row)

    def check_for_entry(self, prev_row, current_row):
        # ... (This method's logic is unchanged) ...
        if not self.lock_manager.verify(): raise LockLostError()
        if self.operational_mode != 'trading':
            self.logger.info(f"Mode is '{self.operational_mode}', skipping new entry check."); return
        signal = self.strategy.get_entry_signal(prev_row, current_row)
        if signal:
            self.logger.info(f"!!! ENTRY SIGNAL DETECTED: {signal} !!!")
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                live_price = ticker['ask'] if signal == 'LONG' else ticker['bid']
                self.logger.info(f"Live price for validation: {live_price}")
                if live_price is None: self.logger.warning("Could not fetch a valid live price. Skipping."); return
                signal_price = current_row['close']; price_difference = abs(live_price - signal_price)
                max_allowed_slippage = MAX_SLIPPAGE_TICKS * self.tick_size
                if price_difference > max_allowed_slippage:
                    self.logger.warning(f"Slippage check failed! Diff > Max Allowed. Skipping."); return
                self.logger.info(f"Slippage check passed.")
                initial_amount = self.trade_size_usd / live_price
                final_amount = max(initial_amount, self.min_trade_amount)
                if final_amount > initial_amount: self.logger.info(f"Size adjusted up to meet exchange minimum.")
                self.trade_size_in_asset = final_amount
                formatted_amount = self.exchange.amount_to_precision(self.symbol, self.trade_size_in_asset)
                self.logger.info(f"Calculated final trade size: {formatted_amount} (Value: ~${self.trade_size_usd:.2f})")
                order = self.exchange.create_market_order(self.symbol, 'buy' if signal == 'LONG' else 'sell', formatted_amount)
                self.in_position, self.position_type, self.entry_price = True, signal, order['price']
                self.entry_time = datetime.now(timezone.utc)
                self.logger.info(f"--- TRADE EXECUTED --- New Position: {self.position_type} @ {self.entry_price}")
                self.state_manager.save_state(self.get_current_state_dict())
            except Exception as e: self.logger.error(f"Error placing order: {e}", exc_info=True); self.in_position = False

    def check_for_exit(self, current_price):
        # ... (This method's logic is unchanged) ...
        if not self.in_position: return
        exit_reason = "Manual Exit" if self.operational_mode == 'exit_all' else None
        if not exit_reason:
            params = self.config['parameters']; stop_loss_pct = params['stop_loss_percent']
            take_profit_pct = stop_loss_pct * params.get('rr_ratio', 2.0)
            if self.position_type == 'LONG':
                if current_price <= self.entry_price * (1 - stop_loss_pct): exit_reason = "Stop-Loss"
                elif current_price >= self.entry_price * (1 + take_profit_pct): exit_reason = "Take-Profit"
            elif self.position_type == 'SHORT':
                if current_price >= self.entry_price * (1 + stop_loss_pct): exit_reason = "Stop-Loss"
                elif current_price <= self.entry_price * (1 - take_profit_pct): exit_reason = "Take-Profit"
        if exit_reason: self.close_position(exit_reason)

    def close_position(self, exit_reason):
        # ... (This method's logic is unchanged) ...
        if not self.in_position: return
        if not self.lock_manager.verify(): raise LockLostError()
        self.logger.info(f"!!! ATTEMPTING TO CLOSE POSITION: {exit_reason} !!!")
        try:
            formatted_amount = self.exchange.amount_to_precision(self.symbol, self.trade_size_in_asset)
            order = self.exchange.create_market_order(self.symbol, 'sell' if self.position_type == 'LONG' else 'buy', formatted_amount)
            exit_price = order['price']; exit_time = datetime.now(timezone.utc)
            pnl = ((exit_price - self.entry_price) if self.position_type == 'LONG' else (self.entry_price - exit_price)) * self.trade_size_in_asset
            self.logger.info(f"--- POSITION CLOSED --- PnL: ${pnl:.2f}")
            trade_record = {'trader_id': self.trader_id, 'instance_id': self.instance_id, 'exchange': self.exchange_id, 'symbol': self.symbol,
                'position_type': self.position_type, 'entry_price': self.entry_price, 'exit_price': exit_price,
                'entry_time': self.entry_time, 'exit_time': exit_time, 'trade_size': self.trade_size_in_asset,
                'pnl': pnl, 'exit_reason': exit_reason}
            self.state_manager.save_trade_history(trade_record)
            self.in_position, self.position_type, self.entry_price, self.trade_size_in_asset, self.entry_time = False, None, 0, 0, None
            self.state_manager.save_state(self.get_current_state_dict())
            if self.operational_mode == 'exit_all':
                self.operational_mode = 'standby'
                self.state_manager.control_collection.update_one({'_id': self.trader_id}, {'$set': {'operational_mode': 'standby'}})
        except Exception as e: self.logger.error(f"Error during position close: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live crypto trading bot.')
    parser.add_argument('--config', type=str, required=True, help='Path to the strategy JSON config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    db_client = None; logger = logging.getLogger(__name__)
    if config:
        try:
            exchange_id = config.get('exchange', 'unknown').lower()
            sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', config.get('symbol', ''))
            trader_id = f"{exchange_id}_{config.get('strategy_name', 'UnknownStrategy')}_{sanitized_symbol}_{config.get('timeframe', '')}"
            logger = setup_logging(trader_id)
            logger.info("Establishing timezone-aware connection to MongoDB...")
            db_client = MongoClient(main_config.MONGO_URI, serverSelectionTimeoutMS=5000, tz_aware=True)
            db_client.server_info()
            logger.info("MongoDB connection successful.")
            trader = Trader(config, db_client, logger)
            trader.run()
        except errors.ConnectionFailure as e: logger.critical(f"Could not connect to MongoDB. Error: {e}")
        except Exception as e: logger.critical(f"Failed to initialize or run Trader. Error: {e}", exc_info=True)
        finally:
            if db_client: db_client.close(); logger.info("MongoDB connection closed.")
