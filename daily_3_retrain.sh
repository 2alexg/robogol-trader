#!/bin/bash

# --- CONFIGURATION ---
PROJECT_DIR="/home/alexgol/Projects/robogol-trader"
WORK_DIR="/home/alexgol/ML_Models"
VENV_PATH="/home/alexgol/python/venv"
LOG_FILE="$WORK_DIR/daily_3_retrain.log"

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

# 1. 15M APT Model (Physics)
train_and_deploy "APT/USDT" "15m" "Physics" 10000 "physics_15m_apt.pkl"

# 2. 15M AVAX Model (Physics)
train_and_deploy "AVAX/USDT" "15m" "Physics" 10000 "physics_15m_avax.pkl"

# 3. 15M ETH Model (Physics)
train_and_deploy "ETH/USDT" "15m" "Physics" 10000 "physics_15m_eth.pkl"

# 4. 15M POL Model (Physics)
train_and_deploy "POL/USDT" "15m" "Physics" 10000 "physics_15m_pol.pkl"

echo "--- [$(date)] Cycle Complete ---" >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"
