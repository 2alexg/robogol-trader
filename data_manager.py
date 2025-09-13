# data_manager.py
#
# Description:
# A reusable data manager that loads and caches OHLCV data from MongoDB.
# This component is designed to be imported by other parts of the trading system,
# such as the optimizer or a live trading engine.
#
# Author: Gemini
# Date: 2025-07-29

import pandas as pd
from pymongo import MongoClient
import config

class DataManager:
    """
    Handles loading and providing all OHLCV data needed for various trading components.
    Data is cached in memory to avoid repeated database calls.
    """
    def __init__(self, symbols, timeframes):
        """
        Initializes the DataManager and pre-loads all specified data.
        :param symbols: A list or set of symbols to load (e.g., ['BTC/USDT']).
        :param timeframes: A list or set of timeframes to load (e.g., ['1d', '1h']).
        """
        print("Initializing Data Manager...")
        self.client = MongoClient(config.MONGO_URI)
        self.db = self.client[config.MONGO_DB_NAME]
        self._data_cache = {}
        self._load_all_data(symbols, timeframes)

    def _load_all_data(self, symbols, timeframes):
        """Pre-loads all symbol/timeframe combinations into a cache."""
        print("--- Pre-loading all data into memory ---")
        for symbol in symbols:
            for timeframe in timeframes:
                cache_key = f"{symbol}_{timeframe}"
                print(f"Loading data for {cache_key}...")
                collection_name = f"{symbol.replace('/', '_')}_{timeframe}"
                
                if collection_name not in self.db.list_collection_names():
                    print(f"Warning: Collection '{collection_name}' not found.")
                    continue

                # Load all data from the collection and sort by timestamp
                data = list(self.db[collection_name].find({}, {'_id': 0}).sort('timestamp', 1))
                if not data:
                    print(f"Warning: No data in collection '{collection_name}'.")
                    continue
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                self._data_cache[cache_key] = df[['open', 'high', 'low', 'close', 'volume']]
                print(f"Loaded {len(df)} records for {cache_key}.")
        print("--- Data loading complete ---\n")

    def get_data(self, symbol, timeframe):
        """
        Returns a copy of the cached DataFrame to prevent accidental modification.
        :param symbol: The symbol to retrieve (e.g., 'BTC/USDT').
        :param timeframe: The timeframe to retrieve (e.g., '1h').
        :return: A pandas DataFrame with the requested OHLCV data.
        """
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in self._data_cache:
            raise ValueError(f"Data for {cache_key} not loaded. Check your configuration.")
        return self._data_cache[cache_key].copy()
