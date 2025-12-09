# data_ingestion_engine.py
#
# Description:
# This script fetches historical OHLCV (Open, High, Low, Close, Volume) data
# from a cryptocurrency exchange for specified pairs and timeframes.
# It uses a decoupled repository pattern to store the data in MongoDB.
# Configuration is loaded from an external config.py file.
#
# Author: Gemini
# Date: 2025-10-18 (v9 - Added batching to prevent BSONObjectTooLarge error)

import ccxt
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from abc import ABC, abstractmethod
import time
import datetime
import os
import argparse  # Added for command-line arguments
import sys         # Added for exiting
import numpy as np # Added for array_split

# --- Configuration ---
# All settings are now loaded from the config.py file.
try:
    import config
except ImportError:
    print("Error: Configuration file 'config.py' not found.")
    print("Please create it in the same directory as this script.")
    sys.exit(1)

# --- FIX: Define a batch size for MongoDB operations ---
MONGO_BATCH_SIZE = 10000


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

    def fetch_all_ohlcv_since(self, symbol, timeframe, since=None, limit=config.CANDLE_LIMIT):
        """
        Fetches ALL OHLCV data for a given symbol and timeframe since a
        specific timestamp, iterating as needed.
        """
        if not self.exchange.has['fetchOHLCV']:
            print(f"Warning: {self.exchange.name} does not support fetchOHLCV.")
            return pd.DataFrame()

        all_candles = []
        fetch_since = since
        timeframe_in_ms = self.exchange.parse_timeframe(timeframe) * 1000
        
        print(f"Fetching all {symbol} ({timeframe}) data... This may take a while.")

        while True:
            try:
                print(f"  Fetching {limit} candles from {datetime.datetime.fromtimestamp(fetch_since / 1000, tz=datetime.timezone.utc) if fetch_since else 'exchange default'}...")
                # Fetch the data from the exchange
                ohlcv_data = self.exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=limit)
                
                if not ohlcv_data:
                    print("  No more new data found.")
                    break  # Exit loop if no data is returned

                all_candles.extend(ohlcv_data)
                
                # Get the timestamp of the last candle and add 1ms to avoid overlap
                last_timestamp = ohlcv_data[-1][0]
                fetch_since = last_timestamp + 1 
                
                # Break if we received fewer candles than the limit, meaning we're at the end
                if len(ohlcv_data) < limit:
                    print("  Received last batch of data.")
                    break
                
                # Be respectful to the exchange's API rate limits
                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.NetworkError as e:
                print(f"Network error while fetching {symbol}: {e}. Retrying in 5s...")
                time.sleep(5)
            except ccxt.ExchangeError as e:
                print(f"Exchange error while fetching {symbol}: {e}")
                break
            except Exception as e:
                print(f"An unexpected error occurred while fetching {symbol}: {e}")
                break
        
        if not all_candles:
            return pd.DataFrame()

        # Convert to a pandas DataFrame for easier manipulation
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert timestamp to a readable datetime format (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        # Remove duplicates, keeping the last entry (most recent data)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        print(f"Fetched a total of {len(df)} unique candles for {symbol} ({timeframe}).")
        return df


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
    This implementation uses MongoDB's Time Series collections for efficiency
    and a "delete-then-insert" pattern to overwrite existing data.
    It processes data in batches to avoid BSON size limits.
    """
    def __init__(self, mongo_uri, db_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        print(f"Connected to MongoDB. Database: '{db_name}'")

    def _ensure_timeseries_collection(self, collection_name):
        """
        Ensures a time series collection exists.
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
        Saves a DataFrame of OHLCV data to MongoDB using a "delete and insert"
        strategy, processed in batches to avoid 16MB BSON limit.
        """
        if df.empty:
            print("No new data to save.")
            return

        collection_name = f"{symbol.replace('/', '_')}_{timeframe}"
        self._ensure_timeseries_collection(collection_name)
        collection = self.db[collection_name]

        # --- FIX: Split DataFrame into manageable chunks ---
        num_chunks = int(np.ceil(len(df) / MONGO_BATCH_SIZE))
        df_chunks = np.array_split(df, num_chunks)
        
        print(f"Total records to save: {len(df)}. Processing in {num_chunks} batch(es) of {MONGO_BATCH_SIZE}...")

        total_deleted = 0
        total_inserted = 0

        for i, batch_df in enumerate(df_chunks):
            if batch_df.empty:
                continue
                
            print(f"  Processing batch {i+1}/{num_chunks} ({len(batch_df)} records)...")
            
            # Prepare records for insertion
            records = batch_df.to_dict('records')
            
            # Get list of timestamps to delete/overwrite
            timestamps_to_overwrite = batch_df['timestamp'].to_list()

            # Add metadata to each record *before* inserting
            for record in records:
                record['metadata'] = {'symbol': symbol, 'timeframe': timeframe}

            try:
                # Step 1: Delete any existing records with these timestamps
                if timestamps_to_overwrite:
                    delete_result = collection.delete_many(
                        {'timestamp': {'$in': timestamps_to_overwrite}}
                    )
                    print(f"    Removed {delete_result.deleted_count} old record(s) for overwrite.")
                    total_deleted += delete_result.deleted_count
                
                # Step 2: Insert the new records
                insert_result = collection.insert_many(records)
                print(f"    Successfully inserted {len(insert_result.inserted_ids)} new/overwritten record(s).")
                total_inserted += len(insert_result.inserted_ids)

            except Exception as e:
                print(f"An unexpected error occurred during batch {i+1} delete/insert: {e}")
                print("    Skipping this batch and continuing...")
        
        print(f"Save operation complete for '{collection_name}':")
        print(f"  Total old records removed: {total_deleted}")
        print(f"  Total new records inserted: {total_inserted}")


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
    
    # --- Add Argument Parsing ---
    parser = argparse.ArgumentParser(description='Crypto OHLCV Data Ingestion Engine.')
    parser.add_argument(
        '--since',
        type=str,
        help='Start date for historical data pull (e.g., "2023-01-01"). Overrides database check.'
    )
    args = parser.parse_args()
    
    cli_since_ms = None
    if args.since:
        try:
            # Convert YYYY-MM-DD string to a UTC timestamp in milliseconds
            dt = datetime.datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            cli_since_ms = int(dt.timestamp() * 1000)
            print(f"--- Manual Backfill Mode: Fetching data since {args.since} ---")
        except ValueError:
            print(f"Error: Invalid date format for --since. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
         print("--- Starting Data Ingestion Engine (Incremental Mode) ---")

    
    # Initialize components using settings from config.py
    fetcher = DataFetcher(config.EXCHANGE)
    repository = MongoRepository(config.OPTIM_MONGO_URI, config.OPTIM_MONGO_DB_NAME)

    # Loop through each symbol and timeframe from the configuration
    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            print("-" * 50)
            print(f"Processing: {symbol} - {timeframe}")
            
            # 1. Determine the final 'since' timestamp to use
            # Prioritize the command-line argument if it was provided
            final_since_ms = None
            if cli_since_ms is not None:
                final_since_ms = cli_since_ms
                print(f"Using manual start date: {args.since}")
            else:
                # Fall back to the old logic: get latest from DB
                latest_timestamp_ms = repository.get_latest_timestamp(symbol, timeframe)
                if latest_timestamp_ms:
                    final_since_ms = latest_timestamp_ms
                    print(f"Resuming from last record in DB: {datetime.datetime.fromtimestamp(latest_timestamp_ms / 1000, tz=datetime.timezone.utc)}")
                else:
                    print("No previous data found. Fetching from the beginning (up to exchange limit).")
            
            # 2. Fetch all new data from the exchange
            ohlcv_df = fetcher.fetch_all_ohlcv_since(symbol, timeframe, since=final_since_ms, limit=config.CANDLE_LIMIT)
            
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

                # Save the new, clean data to the database
                repository.save_ohlcv(ohlcv_df, symbol, timeframe)
            
            # 4. Be respectful to the exchange's API rate limits
            # (Handled inside the fetch_all_ohlcv_since loop)

    print("\n--- Data Ingestion Engine finished its run. ---")


if __name__ == "__main__":
    main()
