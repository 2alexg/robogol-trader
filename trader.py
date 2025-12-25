# trader.py
#
# Description:
# The complete, definitive, and production-ready multi-exchange live
# execution engine. This version delegates exit price calculations
# to the strategy and includes a 'Paper Trading' mode for simulation.
#
# UPDATED: Added 'signal_mode' support to handle "Perfect Loser" strategies
# by reversing signals (Normal <-> Reverse).
#
# Author: Gemini
# Date: 2025-12-20

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
from decimal import Decimal

# --- Core Component Imports ---
from trader_core import MongoLockManager, MongoStateManager, ControlSignalChecker, LockLostError

# --- Main Configuration ---
try:
    import config as main_config
except ImportError:
    logging.basicConfig(level=logging.CRITICAL); logging.critical("FATAL: Could not find 'config.py'.")
    exit(1)

DATA_POLLING_INTERVAL_S = 10; DATA_POLLING_TIMEOUT_S = 120; TRADE_ENTRY_WINDOW_S = 120
DEFAULT_TRADE_SIZE_USD = 20.0; MAX_SLIPPAGE_TICKS = 5; INTRA_CANDLE_CHECK_INTERVAL_S = 30
CONFIRM_ORDER_TIMEOUT_S = 30; CONFIRM_ORDER_POLL_INTERVAL_S = 2

# --- Logging & Helper Functions (Unchanged) ---
def setup_logging(trader_id, paper_trading=False):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    prefix = "[PAPER] " if paper_trading else ""
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - {prefix}[%(trader_id)s] - %(levelname)s - %(message)s',
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
    except (ImportError, AttributeError) as e: return None

class Trader:
    def __init__(self, config, db_client, logger):
        self.config = config; self.symbol = config['symbol']; self.timeframe = config['timeframe']
        self.exchange_id = config['exchange'].lower()
        self.logger = logger
        self.paper_trading = config.get('paper_trading', False)
        
        # --- NEW: Signal Mode (Normal vs Reverse) ---
        # Default is 'normal'. Use 'reverse' to profit from "Perfect Loser" strategies.
        self.signal_mode = config.get('signal_mode', 'normal').lower()

        # --- Check for multi-timeframe requirement ---
        self.high_timeframe = config.get('high_timeframe')
        self.high_tf_df = None # Will hold the high-timeframe data

        sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', self.symbol)
        self.trader_id = f"{self.exchange_id}_{config['strategy_name']}_{sanitized_symbol}_{self.timeframe}"
        if self.paper_trading: self.trader_id += "_PAPER"
        
        self.logger.info(f"Initializing trader with ID: {self.trader_id}")
        
        if self.signal_mode == 'reverse':
             self.logger.info("!!! REVERSE MODE ACTIVE: ALL SIGNALS WILL BE INVERTED !!!")
        
        if self.paper_trading:
            self.logger.info("!!! RUNNING IN PAPER TRADING MODE - NO REAL ORDERS WILL BE EXECUTED !!!")

        if self.high_timeframe:
            self.logger.info(f"Multi-timeframe strategy detected. High TF: {self.high_timeframe}")
            
        self.instance_id = str(uuid.uuid4())
        self.lock_manager = MongoLockManager(db_client, self.trader_id, self.instance_id, self.logger)
        self.state_manager = MongoStateManager(db_client, self.trader_id, self.logger)
        self.settings_collection = db_client[main_config.MONGO_DB_NAME]['trader_settings']
        self.trade_size_usd = config.get('trade_size_usd', DEFAULT_TRADE_SIZE_USD)
        self.max_position_size_usd = config.get('max_position_size_usd', self.trade_size_usd)
        self.operational_mode = 'trading'
        self.control_checker = ControlSignalChecker(db_client, self.trader_id, self, self.logger)
        try:
            credentials = { 'apiKey': os.getenv(f"{self.exchange_id.upper()}_API_KEY"), 'secret': os.getenv(f"{self.exchange_id.upper()}_SECRET_KEY"), }
            if self.exchange_id == 'okx': credentials['password'] = os.getenv('OKX_PASSWORD')
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(credentials)
            self.exchange.options['defaultType'] = 'future'
            # if self.exchange_id in ['binance', 'okx']: self.exchange.set_sandbox_mode(True)
        except AttributeError:
            self.logger.critical(f"Exchange '{self.exchange_id}' is not supported by CCXT."); raise
        self.load_market_data()
        StrategyClass = load_strategy_class(strategy_name=config['strategy_name'])
        if not StrategyClass: raise ValueError("Could not load strategy class.")
        self.strategy = StrategyClass(config['parameters'])
        self.load_and_set_initial_state()
        self.timeframe_in_ms = self.exchange.parse_timeframe(self.timeframe) * 1000

        # --- NEW: Model Hot-Reload Tracking ---
        self.model_path = config['parameters'].get('model_path')
        self.last_model_mtime = 0
        if self.model_path and os.path.exists(self.model_path):
            self.last_model_mtime = os.path.getmtime(self.model_path)
            self.logger.info(f"Monitoring model file for updates: {self.model_path}")
        
    def load_market_data(self):
        try:
            self.logger.info(f"Loading market data for {self.symbol} on {self.exchange_id}...")
            self.exchange.load_markets()
            market = self.exchange.market(self.symbol)

            price_precision_value = market['precision']['price']
            if isinstance(price_precision_value, int):
                self.price_decimal_places = price_precision_value
                self.tick_size = 10 ** -self.price_decimal_places
            elif isinstance(price_precision_value, float):
                self.tick_size = price_precision_value
                self.price_decimal_places = abs(Decimal(str(self.tick_size)).as_tuple().exponent)
            else:
                self.tick_size = 0.0001; self.price_decimal_places = 4
                self.logger.warning("Could not determine price precision method. Using safe defaults.")

            min_amount = market.get('limits', {}).get('amount', {}).get('min')
            if min_amount is None:
                self.logger.warning(f"Market 'min' limit is None. Defaulting min trade size to 0.0.")
                min_trade_amount_in_contracts = 0.0
            else:
                min_trade_amount_in_contracts = float(min_amount)

            c_size = market.get('contractSize')
            if c_size is None:
                self.logger.info(f"Contract size not specified or None. Defaulting to 1.0.")
                self.contract_size = 1.0
            else:
                self.contract_size = float(c_size)

            self.min_trade_amount_in_asset = min_trade_amount_in_contracts * self.contract_size
            self.logger.info(f"Market data loaded: Min Asset Size={self.min_trade_amount_in_asset}, Tick Size={self.tick_size}, Price Decimals={self.price_decimal_places}, Contract Size={self.contract_size}")
        except Exception as e:
            self.logger.error(f"Could not load market data: {e}"); raise
            
    def load_and_set_initial_state(self):
        saved_state = self.state_manager.load_state()
        if saved_state:
            self.in_position = saved_state.get('in_position', False); self.position_type = saved_state.get('position_type')
            self.entry_price = saved_state.get('entry_price', 0); self.trade_size_in_asset = saved_state.get('trade_size_in_asset', 0)
            self.entry_time = saved_state.get('entry_time'); last_ts = saved_state.get('last_candle_timestamp')
            self.last_candle_timestamp = pd.to_datetime(last_ts) if last_ts else None
            self.stop_loss_price = saved_state.get('stop_loss_price'); self.take_profit_price = saved_state.get('take_profit_price')
            self.logger.info(f"State restored. In Position: {self.in_position}")
        else:
            self.in_position=False; self.position_type=None; self.entry_price=0; 
            self.trade_size_in_asset=0; self.last_candle_timestamp=None; self.entry_time=None
            self.stop_loss_price = None; self.take_profit_price = None
            self.logger.info("Initialized with fresh state.")
    def get_current_state_dict(self):
        return { 'in_position': self.in_position, 'position_type': self.position_type, 'entry_price': self.entry_price, 
                 'trade_size_in_asset': self.trade_size_in_asset, 'last_candle_timestamp': self.last_candle_timestamp, 
                 'entry_time': self.entry_time, 'stop_loss_price': self.stop_loss_price, 'take_profit_price': self.take_profit_price }
    def run(self):
        if not self.lock_manager.acquire(): sys.exit(1)
        try:
            self.logger.info("--- Starting Live Trading Engine ---"); self.control_checker.start()
            if self.last_candle_timestamp is None: self.update_latest_closed_candle_data()
            while True:
                # --- NEW: Check for model update at the start of every loop ---
                self.check_for_model_update()
                if self.in_position: self.run_intra_candle_checks()
                else: self.sleep_until_next_candle()
                self.update_latest_closed_candle_data()
        except LockLostError: self.logger.critical("SHUTTING DOWN: Lock lost.")
        except KeyboardInterrupt: self.logger.info("Keyboard interrupt. Shutting down...")
        except Exception as e: self.logger.error(f"Unexpected critical error: {e}. Shutting down.", exc_info=True)
        finally:
            self.control_checker.stop(); self.lock_manager.release()
    def check_for_model_update(self):
        """
        Checks if the model pickle file has been modified.
        If so, re-initializes the strategy to load the new brain.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            return

        try:
            current_mtime = os.path.getmtime(self.model_path)
            if current_mtime > self.last_model_mtime:
                self.logger.info(f"!!! NEW MODEL DETECTED (mtime: {current_mtime}) !!!")
                self.logger.info("Reloading strategy to activate new brain...")
                
                # Re-load the strategy class and re-instantiate
                # This automatically calls joblib.load() in the strategy's __init__
                StrategyClass = load_strategy_class(strategy_name=self.config['strategy_name'])
                self.strategy = StrategyClass(self.config['parameters'])
                
                self.last_model_mtime = current_mtime
                self.logger.info(">>> Strategy Reloaded Successfully. Trading with NEW MODEL. <<<")
        except Exception as e:
            self.logger.error(f"Failed to reload model: {e}")
    def run_intra_candle_checks(self):
        next_candle_time = self.last_candle_timestamp + timedelta(milliseconds=self.timeframe_in_ms)
        self.logger.info(f"In position. Monitoring until {next_candle_time:%H:%M:%S %Z}.")
        while datetime.now(timezone.utc) < next_candle_time:
            if not self.in_position:
                self.logger.info("Position closed. Ending monitoring."); break
            try:
                self.check_for_intra_candle_tp()
            except Exception as e: self.logger.warning(f"Error during intra-candle check: {e}")
            time.sleep(INTRA_CANDLE_CHECK_INTERVAL_S)
        self.logger.info("Intra-candle monitoring finished.")
    
    def update_latest_closed_candle_data(self):
        if not self.lock_manager.verify(): raise LockLostError()
        
        # Fetch high-timeframe data if needed
        if self.high_timeframe:
            self.fetch_high_timeframe_data()
        
        df = self.fetch_latest_candle_data_with_polling()
        if df is not None:
            self.last_candle_timestamp = df.index[-1]; self.state_manager.save_state(self.get_current_state_dict())
            self.check_for_signals_and_manage_position(df)
        else: self.logger.warning("Failed to fetch new data after polling.")
    
    def fetch_latest_candle_data_with_polling(self):
        start_time = time.time(); self.logger.info("Polling for next closed candle...")
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
        
    def fetch_high_timeframe_data(self):
        try:
            self.logger.info(f"Fetching high-timeframe data ({self.high_timeframe})...")
            # Fetch enough candles to satisfy the strategy's EMA (e.g., 200)
            candles = self.exchange.fetch_ohlcv(self.symbol, self.high_timeframe, limit=250)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            self.high_tf_df = df
        except Exception as e:
            self.logger.error(f"Failed to fetch high-timeframe data: {e}")
            self.high_tf_df = None
            
    def sleep_until_next_candle(self):
        if self.last_candle_timestamp is None: return
        now_utc = datetime.now(timezone.utc); next_candle_open_time = self.last_candle_timestamp + timedelta(milliseconds=self.timeframe_in_ms)
        sleep_duration = (next_candle_open_time - now_utc).total_seconds()
        if sleep_duration > 0:
            self.logger.info(f"Next candle at {next_candle_open_time:%H:%M:%S %Z}. Sleeping for {sleep_duration:.2f}s...")
            time.sleep(sleep_duration)
            
    def check_for_signals_and_manage_position(self, df):
        # Pass the high-TF dataframe to the strategy
        _, df_with_indicators = self.strategy.calculate_indicators(self.high_tf_df, df.copy())
        
        # Guard clause to prevent crash if strategy returns None
        if df_with_indicators is None:
            self.logger.warning("Strategy returned no data (perhaps high-TF data is missing or insufficient). Skipping signals check.")
            return
            
        df_with_indicators.dropna(inplace=True);
        if df_with_indicators.empty: return

        prev_row, current_row = df_with_indicators.iloc[-2], df_with_indicators.iloc[-1]

        # 1. Check for Exits (Always check this first)
        if self.in_position:
            self.check_for_candle_close_exit(current_row)
        # 2. Check for Entries (Modified)
        # Logic: Check for entry if we are NOT in position OR if we have room to add
        current_value = self.trade_size_in_asset * current_row['close']
        can_add_to_position = self.in_position and (current_value < self.max_position_size_usd)
        
        if (not self.in_position or can_add_to_position):
            signal_age = (datetime.now(timezone.utc) - current_row.name).total_seconds()
            if signal_age <= TRADE_ENTRY_WINDOW_S:
                self.check_for_entry(prev_row, current_row)

    def _await_order_fill(self, order_id, order_type):
        """
        Waits for order to fill. Returns (filled_amount, average_price).
        If partial fill occurs on timeout/cancel, returns the partial amount.
        Returns (0, None) on total failure.
        """
        start_time = time.time()
        while time.time() - start_time < CONFIRM_ORDER_TIMEOUT_S:
            self.logger.info(f"Waiting for order {order_id} to fill...")
            try:
                if self.exchange_id == 'bybit':
                    try:
                        order_status = self.exchange.fetch_closed_order(order_id, self.symbol)
                        if order_status['status'] == 'closed':
                            self.logger.info(f"Order {order_id} confirmed as filled (closed).")
                            return order_status.get('filled', 0), order_status['average']
                    except ccxt.OrderNotFound:
                        self.logger.debug(f"Order {order_id} not in closed orders, checking open orders.")
                        order_status = self.exchange.fetch_open_order(order_id, self.symbol)
                else:
                    order_status = self.exchange.fetch_order(order_id, self.symbol)
                    if order_status['status'] == 'closed':
                        self.logger.info(f"Order {order_id} confirmed as filled.")
                        return order_status.get('filled', 0), order_status['average']
                
                # If canceled externally, return whatever was filled
                if order_status['status'] == 'canceled':
                    filled = order_status.get('filled', 0)
                    if filled > 0:
                        self.logger.warning(f"Order {order_id} was canceled but partially filled.")
                        return filled, order_status['average']
                    self.logger.warning(f"Order {order_id} was canceled."); return 0, None

            except ccxt.OrderNotFound:
                self.logger.warning(f"Order {order_id} not found yet, will retry.")
            except Exception as e:
                self.logger.warning(f"Could not fetch order status for {order_id}: {e}")
            time.sleep(CONFIRM_ORDER_POLL_INTERVAL_S)
            
        if order_type == 'limit':
            self.logger.warning(f"Order {order_id} did not fill in time. Canceling.")
            try:
                self.exchange.cancel_order(order_id, self.symbol)
                self.logger.info(f"Successfully canceled order {order_id}.")
                
                # Fetch final state to check for partial fills
                final_status = self.exchange.fetch_order(order_id, self.symbol)
                filled = final_status.get('filled', 0)
                if filled > 0:
                    self.logger.info(f"Partial fill detected on timeout/cancel: {filled} contracts.")
                    return filled, final_status['average']
                    
            except Exception as e:
                self.logger.error(f"Failed to cancel order {order_id}. MANUAL INTERVENTION REQUIRED! Error: {e}")
        else:
            self.logger.error(f"MARKET order {order_id} did not fill in time. MANUAL INTERVENTION REQUIRED!")
        return 0, None
            
    def check_for_entry(self, prev_row, current_row):
        if not self.lock_manager.verify(): raise LockLostError()
        if self.operational_mode != 'trading':
            self.logger.info(f"Mode is '{self.operational_mode}', skipping entry check."); return
        
        signal = self.strategy.get_entry_signal(prev_row, current_row)

        # --- NEW: Reverse Signal Logic ---
        # If signal_mode is 'reverse', we flip the signal BEFORE proceeding.
        if signal and self.signal_mode == 'reverse':
            original_signal = signal
            if signal == 'LONG': signal = 'SHORT'
            elif signal == 'SHORT': signal = 'LONG'
            self.logger.info(f"Reverse Mode: Inverted {original_signal} signal to {signal}.")
        
        if signal:
            # --- NEW: Pyramiding Checks ---
            if self.in_position:
                # 1. Direction Check: Only add if signal matches current position
                if signal != self.position_type:
                    self.logger.info(f"Ignored {signal} signal because we are currently {self.position_type}.")
                    return
                
                # 2. Cap Check: Ensure we don't exceed max size
                current_price = current_row['close']
                current_exposure = self.trade_size_in_asset * current_price
                
                # If adding one more trade exceeds max, abort (or you could adjust size to fill remainder)
                if current_exposure + self.trade_size_usd > self.max_position_size_usd:
                    self.logger.info("Max position size reached. Skipping pyramiding entry.")
                    return

                self.logger.info(f"Pyramiding: Adding to {self.position_type} position. Current Size: ${current_exposure:.2f}")

            self.logger.info(f"!!! ENTRY SIGNAL DETECTED: {signal} !!!")
            try:
                signal_price = current_row['close']
                limit_price = self.exchange.price_to_precision(self.symbol, signal_price)
                desired_amount_in_asset = self.trade_size_usd / float(limit_price)
                final_amount_in_asset = max(desired_amount_in_asset, self.min_trade_amount_in_asset)
                if final_amount_in_asset > desired_amount_in_asset: self.logger.info(f"Size adjusted up to meet exchange minimum.")
                quantity_in_contracts = final_amount_in_asset / self.contract_size
                formatted_amount = self.exchange.amount_to_precision(self.symbol, quantity_in_contracts)
                self.logger.info(f"Calculated final order: {formatted_amount} contracts (True Size: {final_amount_in_asset:.8f})")
                
                if self.paper_trading:
                    # --- PAPER TRADING SIMULATION ---
                    self.logger.info(f"PAPER TRADING: Simulating ENTRY order of {formatted_amount} contracts at {limit_price}...")
                    filled_qty = final_amount_in_asset   # Assume full fill
                    confirmed_price = float(limit_price) # Assume fill at limit price
                else:
                    # --- REAL TRADING EXECUTION ---
                    order_params = {};
                    if self.exchange_id == 'okx': order_params['posSide'] = 'long' if signal == 'LONG' else 'short'
                    self.logger.info(f"Placing LIMIT order at signal price: {limit_price}...")
                    order = self.exchange.create_limit_order(self.symbol, 'buy' if signal == 'LONG' else 'sell', float(formatted_amount), float(limit_price), params=order_params)
                    order_id = order['id']
                    
                    filled_qty, confirmed_price = self._await_order_fill(order_id, 'limit')
                
                if filled_qty == 0 or confirmed_price is None:
                    self.logger.error("Could not confirm trade execution (No fill). Aborting entry."); return

                new_position = False
                if self.in_position:
                    # Recalculate Weighted Average Entry Price
                    total_cost_old = self.trade_size_in_asset * self.entry_price
                    total_cost_new = filled_qty * confirmed_price
                    new_total_qty = self.trade_size_in_asset + filled_qty

                    self.entry_price = (total_cost_old + total_cost_new) / new_total_qty
                    self.trade_size_in_asset = new_total_qty

                    self.logger.info(f"Position Increased. New Avg Entry: {self.entry_price:.4f}, Total Size: {self.trade_size_in_asset}")
                else:
                    # Update size to actual filled size in case of partial fill
                    if filled_qty != self.trade_size_in_asset:
                        self.logger.info(f"Adjusting position size from {self.trade_size_in_asset} to filled amount: {filled_qty}")
                    # Standard New Position
                    self.in_position, self.position_type, self.entry_price = True, signal, confirmed_price
                    self.trade_size_in_asset = filled_qty
                    self.entry_time = datetime.now(timezone.utc)
                    new_position = True
                
                # --- CRITICAL: Update SL/TP for the WHOLE position ---
                # We recalculate SL/TP based on the NEW Average Entry Price and CURRENT volatility.
                # Note: 'signal' here is already reversed if needed, so SL/TP will be calculated correctly for the new direction.
                self.stop_loss_price, self.take_profit_price = self.strategy.calculate_exit_prices(
                    entry_price=self.entry_price, 
                    signal=signal, 
                    current_row=current_row
                )

                self.logger.info(f"--- TRADE CONFIRMED --- New Position: {self.position_type} @ {self.entry_price}")
                if new_position:
                    self.logger.info(f"SL: {self.stop_loss_price:.{self.price_decimal_places}f}, TP: {self.take_profit_price:.{self.price_decimal_places}f}")
                else:
                    self.logger.info(f"SL/TP Updated: SL: {self.stop_loss_price:.{self.price_decimal_places}f}, TP: {self.take_profit_price:.{self.price_decimal_places}f}")
                self.state_manager.save_state(self.get_current_state_dict())
            except Exception as e:
                self.logger.error(f"Error placing order: {e}", exc_info=True)
                # Handle rollback if needed (simplest is to just not update state if order failed)            

    def check_for_candle_close_exit(self, current_row):
        if not self.in_position: return
        exit_reason = None; current_price = current_row['close']
        if self.operational_mode == 'exit_all': exit_reason = "Manual Exit"
        elif self.position_type == 'LONG':
            if current_price <= self.stop_loss_price: exit_reason = "Stop-Loss"
            elif current_price >= self.take_profit_price: exit_reason = "Take-Profit"
        elif self.position_type == 'SHORT':
            if current_price >= self.stop_loss_price: exit_reason = "Stop-Loss"
            elif current_price <= self.take_profit_price: exit_reason = "Take-Profit"
        if exit_reason: self.close_position(exit_reason)
        
    def check_for_intra_candle_tp(self):
        if not self.in_position or self.position_type is None: return
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            if price is None: return
            self.logger.debug(f"Intra-candle check: Live price={price}, TP Target={self.take_profit_price}")
            if self.position_type == 'LONG' and price >= self.take_profit_price:
                self.logger.info("Intra-candle Take-Profit hit for LONG position!")
                self.close_position("Take-Profit")
            elif self.position_type == 'SHORT' and price <= self.take_profit_price:
                self.logger.info("Intra-candle Take-Profit hit for SHORT position!")
                self.close_position("Take-Profit")
        except Exception as e: self.logger.warning(f"Could not perform intra-candle TP check: {e}")
        
    def close_position(self, exit_reason):
        if not self.in_position: return
        if not self.lock_manager.verify(): raise LockLostError()
        self.logger.info(f"!!! ATTEMPTING TO CLOSE POSITION: {exit_reason} !!!")
        try:
            quantity_in_contracts = self.trade_size_in_asset / self.contract_size
            formatted_amount = self.exchange.amount_to_precision(self.symbol, quantity_in_contracts)
            order_type = 'limit' if exit_reason == "Take-Profit" else 'market'
            
            if self.paper_trading:
                # --- PAPER TRADING SIMULATION ---
                self.logger.info(f"PAPER TRADING: Simulating EXIT ({order_type})...")
                filled_qty = self.trade_size_in_asset # Assume full fill
                
                if order_type == 'limit':
                    # For limit (TP), we assume we exited exactly at our target price
                    confirmed_exit_price = self.take_profit_price
                else:
                    # For market (SL/Manual), we fetch the current last price to simulate market fill
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    confirmed_exit_price = ticker['last']
                    
                self.logger.info(f"PAPER TRADING: Simulated exit price: {confirmed_exit_price}")
            
            else:
                # --- REAL TRADING EXECUTION ---
                order_params = {};
                if self.exchange_id == 'okx': order_params['posSide'] = 'long' if self.position_type == 'LONG' else 'short'
                
                if order_type == 'limit':
                    exit_price_target = self.exchange.price_to_precision(self.symbol, self.take_profit_price)
                    self.logger.info(f"Placing LIMIT order at {exit_price_target}")
                    order = self.exchange.create_limit_order(self.symbol, 'sell' if self.position_type == 'LONG' else 'buy', float(formatted_amount), float(exit_price_target), params=order_params)
                else:
                    self.logger.info("Placing MARKET order for immediate exit.")
                    order = self.exchange.create_market_order(self.symbol, 'sell' if self.position_type == 'LONG' else 'buy', float(formatted_amount), params=order_params)
                
                filled_qty, confirmed_exit_price = self._await_order_fill(order['id'], order_type)
            
            if filled_qty == 0 or confirmed_exit_price is None:
                self.logger.error(f"Could not confirm exit. Position may still be open.")
                if order_type == 'limit': return
                return
            
            exit_time = datetime.now(timezone.utc)
            # Use actual filled_qty for PnL calculation
            pnl = ((confirmed_exit_price - self.entry_price) if self.position_type == 'LONG' else (self.entry_price - confirmed_exit_price)) * filled_qty
            self.logger.info(f"--- POSITION CLOSED --- PnL: ${pnl:.2f}")
            trade_record = {'trader_id': self.trader_id, 'instance_id': self.instance_id, 'exchange': self.exchange_id, 'symbol': self.symbol, 'position_type': self.position_type, 'entry_price': self.entry_price, 'exit_price': confirmed_exit_price, 'entry_time': self.entry_time, 'exit_time': exit_time, 'trade_size': filled_qty, 'pnl': pnl, 'exit_reason': exit_reason, 'is_paper': self.paper_trading}
            self.state_manager.save_trade_history(trade_record)
            self.in_position, self.position_type, self.entry_price, self.trade_size_in_asset, self.entry_time = False, None, 0, 0, None
            self.stop_loss_price, self.take_profit_price = None, None
            self.state_manager.save_state(self.get_current_state_dict())
            if self.operational_mode == 'exit_all':
                self.operational_mode = 'standby'
                self.state_manager.db['trader_controls'].update_one({'_id': self.trader_id}, {'$set': {'operational_mode': 'standby'}})
        except Exception as e: self.logger.error(f"Error during position close: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live crypto trading bot.')
    parser.add_argument('--config', type=str, required=True, help='Path to the strategy JSON config file.')
    parser.add_argument('--paper-trading', action='store_true', help='Enable paper trading mode (simulation only).')
    args = parser.parse_args()
    config = load_config(args.config)
    db_client = None; logger = logging.getLogger(__name__)
    if config:
        try:
            # Inject paper trading flag into config
            config['paper_trading'] = args.paper_trading
            
            exchange_id = config.get('exchange', 'unknown').lower()
            sanitized_symbol = re.sub(r'[^a-zA-Z0-9]', '', config.get('symbol', ''))
            trader_id = f"{exchange_id}_{config.get('strategy_name', 'UnknownStrategy')}_{sanitized_symbol}_{config.get('timeframe', '')}"
            if args.paper_trading: trader_id += "_PAPER"
            
            logger = setup_logging(trader_id, args.paper_trading)
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
