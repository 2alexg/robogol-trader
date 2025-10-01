# trader_core.py
#
# Description:
# A module containing the core, reusable components for the trading platform,
# including the distributed lock manager, the persistent state manager, and
# the high-responsiveness control signal checker.
#
# Author: Gemini
# Date: 2025-09-26

import time
import logging
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient, errors
from threading import Thread, Event

try:
    import config as main_config
except ImportError:
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical("FATAL: Could not find 'config.py'. Please ensure it exists.")
    exit(1)

# --- Constants ---
LOCK_HEARTBEAT_INTERVAL_S = 10
LOCK_EXPIRY_S = 30 
CONTROL_POLLING_INTERVAL_S = 10

# --- Custom Exception ---
class LockLostError(Exception):
    """Custom exception to signal that lock ownership has been lost."""
    pass

# --- Distributed Lock Manager ---
class MongoLockManager:
    """Handles acquiring and maintaining a distributed lock in MongoDB."""
    def __init__(self, db_client, trader_id, instance_id, logger):
        self.trader_id = trader_id
        self.instance_id = instance_id
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
                self.logger.warning(f"Found stale lock. Taking over.")
                self.collection.update_one({'_id': self.trader_id}, update_doc)
                self._start_heartbeat()
                return True
            else:
                self.logger.error(f"Another instance ({lock_doc.get('instance_id')}) is running. Aborting.")
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
        try:
            lock_doc = self.collection.find_one({'_id': self.trader_id})
            return lock_doc and lock_doc.get('instance_id') == self.instance_id
        except Exception:
            return False 

    def release(self):
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
                    self.logger.critical("Heartbeat failed: Lock taken by another instance!")
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
            self.logger.debug(f"State saved.")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load_state(self):
        try:
            state = self.state_collection.find_one({'_id': self.trader_id})
            if state:
                self.logger.info(f"Found existing state.")
            else:
                self.logger.info(f"No state found.")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return None

    def save_trade_history(self, trade_data):
        try:
            self.history_collection.insert_one(trade_data)
            self.logger.info("Trade history record saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")

# --- High-Responsiveness Control ---
class ControlSignalChecker:
    """A background thread that continuously polls for control signals."""
    def __init__(self, db_client, trader_id, trader_instance, logger):
        self.db = db_client[main_config.MONGO_DB_NAME]
        self.control_collection = self.db['trader_controls']
        self.trader_id = trader_id
        self.trader = trader_instance
        self.logger = logger
        self._stop_event = Event()
        self._thread = Thread(target=self.run, daemon=True)

    def start(self):
        self.logger.info("Control signal checker thread started.")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.logger.info("Control signal checker thread stopped.")

    def run(self):
        while not self._stop_event.is_set():
            try:
                control_doc = self.control_collection.find_one({'_id': self.trader_id})
                if control_doc:
                    new_mode = control_doc.get('operational_mode', 'trading')
                    if new_mode != self.trader.operational_mode:
                        self.logger.info(f"CONTROL SIGNAL DETECTED: Changing mode to '{new_mode}'.")
                        self.trader.operational_mode = new_mode
            except Exception as e:
                self.logger.error(f"Error in control signal checker: {e}")
            time.sleep(CONTROL_POLLING_INTERVAL_S)
