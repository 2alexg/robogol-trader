# data_ingestion_engine.py
#
# Description:
# This script fetches historical OHLCV (Open, High, Low, Close, Volume) data
# from a cryptocurrency exchange for specified pairs and timeframes.
# It uses a decoupled repository pattern to store the data in MongoDB.
# Configuration is loaded from an external config.py file.
#
# Author: Gemini
# Date: 2025-07-26 (v6 - Externalized configuration)

import ccxt
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from abc import ABC, abstractmethod
import time
import datetime
import os

# --- Configuration ---
# All settings are now loaded from the config.py file.
try:
    import config
except ImportError:
    print("Error: Configuration file 'config.py' not found.")
    print("Please create it in the same directory as this script.")
    exit()


# --- 1. Data Fetcher ---
# Responsible for connecting to the exchange and fetching data.
# It is completely independent of the database.

class DataFetcher:
    """
    Handles fetching OHLCV data from a cryptocurrency exchange using CCXT.
    """
    def __init__(self, exchange_id):
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class()
            # Load markets to access timeframe data
            self.exchange.load_markets()
            print(f"Successfully connected to {self.exchange.name}")
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {exchange_id}: {e}")

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=config.CANDLE_LIMIT):
        """
        Fetches OHLCV data for a given symbol and timeframe.
        """
        if not self.exchange.has['fetchOHLCV']:
            print(f"Warning: {self.exchange.name} does not support fetchOHLCV.")
            return []

        try:
            print(f"Fetching {symbol} ({timeframe}) data...")
            # Fetch the data from the exchange
            ohlcv_data = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv_data:
                print(f"No new data found for {symbol} ({timeframe}).")
                return pd.DataFrame()

            # Convert to a pandas DataFrame for easier manipulation
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Convert timestamp to a readable datetime format (UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            print(f"Fetched {len(df)} candles for {symbol} ({timeframe}).")
            return df
        except ccxt.NetworkError as e:
            print(f"Network error while fetching {symbol}: {e}")
        except ccxt.ExchangeError as e:
            print(f"Exchange error while fetching {symbol}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while fetching {symbol}: {e}")
        return pd.DataFrame()


# --- 2. Data Repository (Abstract Base Class) ---
# This defines the "contract" for any database implementation.

class DataRepository(ABC):
    """
    Abstract base class for data storage repositories.
    """
    @abstractmethod
    def save_ohlcv(self, df, symbol, timeframe):
        pass

    @abstractmethod
    def get_latest_timestamp(self, symbol, timeframe):
        pass

# --- 3. MongoDB Repository Implementation ---
# This is the concrete implementation for MongoDB.

class MongoRepository(DataRepository):
    """
    Manages storing and retrieving data from a MongoDB database.
    This implementation uses MongoDB's Time Series collections for efficiency.
    """
    def __init__(self, mongo_uri, db_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        print(f"Connected to MongoDB. Database: '{db_name}'")

    def _ensure_timeseries_collection(self, collection_name):
        """
        Ensures a time series collection exists.
        NOTE: Unique indexes are not supported, so we don't create one.
        """
        try:
            # Attempt to create the collection
            self.db.create_collection(
                collection_name,
                timeseries={
                    'timeField': 'timestamp',
                    'metaField': 'metadata',
                    'granularity': 'minutes'
                }
            )
            print(f"Created new time series collection: '{collection_name}'")
        except CollectionInvalid:
            # This error means the collection already exists, which is fine.
            pass

    def save_ohlcv(self, df, symbol, timeframe):
        """
        Saves a DataFrame of OHLCV data to the appropriate MongoDB collection.
        Assumes the DataFrame contains only new, non-duplicate data.
        """
        if df.empty:
            print("No new data to save.")
            return

        collection_name = f"{symbol.replace('/', '_')}_{timeframe}"
        self._ensure_timeseries_collection(collection_name)
        collection = self.db[collection_name]

        # Prepare records for insertion
        records = df.to_dict('records')
        # Add metadata to each record
        for record in records:
            record['metadata'] = {'symbol': symbol, 'timeframe': timeframe}

        try:
            collection.insert_many(records)
            print(f"Successfully inserted {len(records)} new records into '{collection_name}'.")
        except Exception as e:
            print(f"An unexpected error occurred during bulk insert: {e}")


    def get_latest_timestamp(self, symbol, timeframe):
        """
        Retrieves the most recent timestamp for a given symbol and timeframe
        to know where to resume fetching from.
        """
        collection_name = f"{symbol.replace('/', '_')}_{timeframe}"
        if collection_name not in self.db.list_collection_names():
            return None # No data exists yet

        collection = self.db[collection_name]
        latest_record = collection.find_one(sort=[('timestamp', -1)])

        if latest_record:
            # Return timestamp in milliseconds for CCXT 'since' parameter
            return int(latest_record['timestamp'].timestamp() * 1000)
        return None


# --- 4. Main Ingestion Logic ---
# This orchestrates the fetching and storing process.

def main():
    """
    Main function to run the data ingestion engine.
    """
    print("--- Starting Data Ingestion Engine ---")
    
    # Initialize components using settings from config.py
    fetcher = DataFetcher(config.EXCHANGE)
    repository = MongoRepository(config.MONGO_URI, config.MONGO_DB_NAME)

    # Loop through each symbol and timeframe from the configuration
    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            print("-" * 50)
            print(f"Processing: {symbol} - {timeframe}")
            
            # 1. Find the last entry to avoid re-downloading everything
            latest_timestamp_ms = repository.get_latest_timestamp(symbol, timeframe)
            since = None
            if latest_timestamp_ms:
                since = latest_timestamp_ms
                print(f"Last record found at: {datetime.datetime.fromtimestamp(latest_timestamp_ms / 1000, tz=datetime.timezone.utc)}")
            else:
                print("No previous data found. Fetching from the beginning (up to limit).")

            # 2. Fetch new data from the exchange
            ohlcv_df = fetcher.fetch_ohlcv(symbol, timeframe, since=since)
            
            # 3. Process and save the new data
            if not ohlcv_df.empty:
                # *** PREVENT STORING INCOMPLETE CANDLE ***
                # The last candle from the exchange is often the current, live, incomplete candle.
                # We only want to store closed candles to avoid acting on bad data.
                try:
                    # Use the ccxt helper function to get the timeframe duration in seconds
                    timeframe_seconds = fetcher.exchange.parse_timeframe(timeframe)
                    timeframe_duration = pd.to_timedelta(timeframe_seconds, unit='s')
                    
                    last_candle_timestamp = ohlcv_df.iloc[-1]['timestamp']
                    expected_close_time = last_candle_timestamp + timeframe_duration
                    now_utc = datetime.datetime.now(datetime.timezone.utc)

                    if now_utc < expected_close_time:
                        ohlcv_df = ohlcv_df.iloc[:-1]
                        print(f"Removed 1 live/incomplete candle before processing.")
                except Exception as e:
                    print(f"Could not check for incomplete candle: {e}")

                # *** APPLICATION-SIDE DE-DUPLICATION ***
                # If we have a latest timestamp, filter the fetched data to ensure
                # we only insert candles that are strictly newer.
                if latest_timestamp_ms:
                    latest_datetime_utc = pd.to_datetime(latest_timestamp_ms, unit='ms', utc=True)
                    original_count = len(ohlcv_df)
                    ohlcv_df = ohlcv_df[ohlcv_df['timestamp'] > latest_datetime_utc]
                    new_count = len(ohlcv_df)
                    if new_count < original_count:
                        print(f"Removed {original_count - new_count} overlapping candle(s) before saving.")

                # Save the new, clean, de-duplicated data to the database
                repository.save_ohlcv(ohlcv_df, symbol, timeframe)
            
            # 4. Be respectful to the exchange's API rate limits
            time.sleep(fetcher.exchange.rateLimit / 1000)

    print("\n--- Data Ingestion Engine finished its run. ---")


if __name__ == "__main__":
    main()
