#!/bin/bash

# --- CONFIGURATION ---
PROJECT_DIR="/home/alexgol/Projects/robogol-trader"
WORK_DIR="/home/alexgol/ML_Models"
VENV_PATH="/home/alexgol/python/venv"
LOG_FILE="$WORK_DIR/daily_2_retrain.log"

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
    local TEMP_MODEL="$WORK_DIR/temp_$MODEL_FILENAME"

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
        mv "$TEMP_MODEL" "$WORK_DIR/$MODEL_FILENAME"
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

# 1. 5M BTC Model (Physics)
train_and_deploy "BTC/USDT" "5m" "Physics" 10000 "physics_5m_btc.pkl"

# 2. 5M ETH Model (Physics)
train_and_deploy "ETH/USDT" "5m" "Physics" 10000 "physics_5m_eth.pkl"

# 3. 5M LINK Model (Physics)
train_and_deploy "LINK/USDT" "5m" "Physics" 10000 "physics_5m_link.pkl"

# 4. 5M TRX Model (Physics)
train_and_deploy "TRX/USDT" "5m" "Physics" 10000 "physics_5m_trx.pkl"

# 5. 5M BTC Model (VWAP)
train_and_deploy "BTC/USDT" "5m" "Vwap" 10000 "vwap_5m_btc.pkl"

# 6. 5M ETH Model (VWAP)
train_and_deploy "ETH/USDT" "5m" "Vwap" 10000 "vwap_5m_eth.pkl"

# 7. 5M TRX Model (VWAP)
train_and_deploy "TRX/USDT" "5m" "Vwap" 10000 "vwap_5m_trx.pkl"

# 8. 5M UNI Model (VWAP)
train_and_deploy "UNI/USDT" "5m" "Vwap" 10000 "vwap_5m_uni.pkl"

# 9. 5M XRP Model (VWAP)
train_and_deploy "XRP/USDT" "5m" "Vwap" 10000 "vwap_5m_xrp.pkl"

# 10. 5M ZEC Model (VWAP)
train_and_deploy "ZEC/USDT" "5m" "Vwap" 10000 "vwap_5m_zec.pkl"

echo "--- [$(date)] Cycle Complete ---" >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"
