#!/bin/bash

# --- CONFIGURATION ---
PROJECT_DIR="/home/alexgol/Projects/robogol-trader"
VENV_PATH="/home/alexgol/python/venv"
LOG_FILE="$PROJECT_DIR/retrain.log"

# Navigate to project
cd "$PROJECT_DIR" || exit 1
source "$VENV_PATH/bin/activate"

echo "========================================================" >> "$LOG_FILE"
echo "--- [$(date)] Starting Weekly Retraining Cycle ---" >> "$LOG_FILE"

# ---------------------------------------------------------
# STEP 1: Ingest Fresh Data (Runs once for ALL symbols)
# ---------------------------------------------------------
echo "Step 1: Ingesting fresh data for all configured symbols..." >> "$LOG_FILE"
python3 data_ingestion_engine.py >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "CRITICAL WARNING: Data ingestion failed or had errors. Proceeding with caution." >> "$LOG_FILE"
fi

# ---------------------------------------------------------
# FUNCTION: Train and Swap
# ---------------------------------------------------------
train_and_deploy() {
    local SYMBOL=$1
    local TIMEFRAME=$2
    local STRATEGY=$3
    local LIMIT=$4
    local MODEL_FILENAME=$5

    echo "------------------------------------------------" >> "$LOG_FILE"
    echo "Processing: $SYMBOL ($TIMEFRAME) - $STRATEGY" >> "$LOG_FILE"

    # Define temp file
    local TEMP_MODEL="temp_$MODEL_FILENAME"

    # Train Model
    echo "  > Training new model..." >> "$LOG_FILE"
    python3 train_deployable_model.py \
        --symbol "$SYMBOL" \
        --timeframe "$TIMEFRAME" \
        --strategy "$STRATEGY" \
        --limit "$LIMIT" \
        --output "$TEMP_MODEL" >> "$LOG_FILE" 2>&1

    # Atomic Swap
    if [ -f "$TEMP_MODEL" ]; then
        mv "$TEMP_MODEL" "$MODEL_FILENAME"
        echo "  > SUCCESS: Model replaced ($MODEL_FILENAME)." >> "$LOG_FILE"
    else
        echo "  > FAILURE: Training script did not produce a model file for $SYMBOL." >> "$LOG_FILE"
    fi
}

# ---------------------------------------------------------
# STEP 2: Train Your Models
# Add as many lines here as you need!
# ---------------------------------------------------------

# Syntax: train_and_deploy "SYMBOL" "TIMEFRAME" "STRATEGY" LIMIT "OUTPUT_FILE"

# 1. BTC Model (Physics)
train_and_deploy "BTC/USDT" "5m" "Physics" 10000 "physics_btc_5m.pkl"

# 2. ETH Model (Physics)
train_and_deploy "ETH/USDT" "5m" "Physics" 10000 "physics_eth_5m.pkl"

# 3. SOL Model (VWAP)
train_and_deploy "SOL/USDT" "15m" "Vwap" 8000 "vwap_sol_15m.pkl"


echo "--- [$(date)] Cycle Complete ---" >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"