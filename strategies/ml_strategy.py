# strategies/ml_strategy.py
import pandas as pd
import joblib
import numpy as np
from .base_strategy import BaseStrategy

class MLStrategy(BaseStrategy):
    """
    Abstract Base Class for all Machine Learning based strategies.
    Handles Model Loading and Prediction Signals.
    """
    def __init__(self, params):
        super().__init__(params)
        self.model_path = params.get('model_path')
        self.threshold = params.get('threshold', 0.85)
        self.features = [] # To be defined by child classes
        
        # Load the Brain
        if self.model_path:
            self.brain = joblib.load(self.model_path)
            self.model_short = self.brain['short_model']
            self.model_long = self.brain['long_model']
            # Optional: Override threshold from pickle if not in params
            if 'short_threshold' in self.brain:
                self.threshold_short = self.brain['short_threshold']
                self.threshold_long = self.brain['long_threshold']
            else:
                self.threshold_short = self.threshold
                self.threshold_long = self.threshold
        else:
            print("WARNING: No model path provided. Running in Feature-Only mode.")

    def add_features(self, df):
        """
        Pure Abstract Method: Child classes MUST implement this.
        This is where 'Vwap' vs 'Standard' logic lives.
        """
        raise NotImplementedError("Child strategy must implement add_features()")

    def calculate_indicators(self, df_htf, df_ltf):
        """
        Standard interface for trader.py.
        Just calls our custom add_features().
        """
        # We only care about the Low Timeframe (df_ltf) for this ML model
        df_featured = self.add_features(df_ltf.copy())
        return None, df_featured

    def get_entry_signal(self, prev_row, current_row):
        """
        Standard interface for trader.py.
        Extracts features -> Predicts -> Returns 'LONG'/'SHORT'.
        """
        # 1. Prepare Features (1-row DataFrame)
        # We need to reshape the single row back into a DataFrame to match training shape
        features_row = pd.DataFrame([current_row[self.features]])
        
        # 2. Get Probabilities
        prob_short = self.model_short.predict_proba(features_row)[0][1]
        prob_long = self.model_long.predict_proba(features_row)[0][1]
        
        # 3. Decision
        if prob_short >= self.threshold_short:
            return 'SHORT'
        elif prob_long >= self.threshold_long:
            return 'LONG'
        
        return None
    