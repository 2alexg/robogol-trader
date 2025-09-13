# data_manager.py
#
# Description:
# Handles loading and caching of historical market data from MongoDB.
# It is designed to load all necessary data into memory once to provide
# fast access for backtesting and optimization.
#
# Author: Gemini
# Date: 2025-09-13 (v3 - Corrected config parameter name)

import pandas as pd
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure
import config

class DataManager:
    """
    Manages loading and caching of historical OHLCV data from MongoDB.
    """
    def __init__(self, symbols, timeframes):
        """
        Initializes the DataManager, connects to MongoDB, and loads all
        required data into an in-memory cache.
        """
        self.client = None
        self.db = None
        self.data_cache = {}
        try:
            self.client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.server_info() # Trigger exception if cannot connect
            # --- FIX: Changed DATABASE_NAME back to MONGO_DB_NAME ---
            self.db = self.client[config.MONGO_DB_NAME]
            print("Successfully connected to MongoDB.")
            self._load_all_data(symbols, timeframes)
        except ConnectionFailure:
            print(f"Error: Could not connect to MongoDB at {config.MONGO_URI}.")
            print("Please ensure MongoDB is running and accessible.")
            exit(1)
        except AttributeError:
             print(f"Error: Could not find 'MONGO_DB_NAME' in your config.py file.")
             print("Please ensure the database name variable is correctly set.")
             exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during DataManager initialization: {e}")
            exit(1)

    def _load_all_data(self, symbols, timeframes):
        """
        Loads all specified symbol/timeframe combinations from the database
        into the self.data_cache dictionary.
        """
        print(f"Loading data for {len(symbols) * len(timeframes)} collection(s)...")
        for symbol in symbols:
            for timeframe in timeframes:
                collection_name = f"{symbol.replace('/', '_')}_{timeframe}"
                collection = self.db[collection_name]
                
                if collection.count_documents({}) > 0:
                    print(f"  Loading {collection_name}...")
                    cursor = collection.find({}).sort('timestamp', DESCENDING)
                    df = pd.DataFrame(list(cursor))
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    self.data_cache[f"{symbol}_{timeframe}"] = df
                else:
                    print(f"  Warning: Collection '{collection_name}' not found or is empty.")
        print("Data loading complete.")

    def get_data(self, symbol, timeframe):
        """
        Retrieves a dataframe from the cache for a given symbol and timeframe.
        Always returns a COPY of the dataframe to ensure data isolation.
        """
        key = f"{symbol}_{timeframe}"
        df = self.data_cache.get(key)
        if df is not None:
            return df.copy() # Return a copy, not a reference
        return None

    def close(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

