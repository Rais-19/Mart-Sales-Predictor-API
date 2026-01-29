"""
Simple Prediction Service for Mart Sales
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from schemas.prediction import SalesInput

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_path: str = "model/mart_sales_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load the XGBoost model saved from the notebook"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at: {self.model_path}")

            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Try to get feature names from the booster
            try:
                self.feature_names = self.model.get_booster().feature_names
                logger.info(f"Model loaded – {len(self.feature_names)} features")
            except:
                self.feature_names = None
                logger.warning("Could not read feature names from model")

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def preprocess(self, input_data: SalesInput) -> pd.DataFrame:
        # Convert to single-row DataFrame
        row = input_data.dict()
        df = pd.DataFrame([row])

        # Apply transformations
        df['Outlet_Age'] = 2026 - df['Outlet_Establishment_Year']
        df = df.drop(columns=['Outlet_Establishment_Year'], errors='ignore')

        # Encode Item_Fat_Content (0/1)
        fat_map = {'Low Fat': 0, 'Regular': 1}
        df['Item_Fat_Content'] = df['Item_Fat_Content'].map(fat_map)

        # Encode Outlet_Size (0/1/2)
        size_map = {'Small': 0, 'Medium': 1, 'High': 2}
        df['Outlet_Size'] = df['Outlet_Size'].map(size_map)

        # One-hot encode remaining categoricals
        cat_cols = ['Item_Type', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type']
        df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)

        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0)
        else:
            logger.warning("No feature names available → alignment may be incomplete")

        logger.debug(f"Processed input columns: {list(df.columns)}")
        return df

    def predict(self, input_data: SalesInput) -> Dict[str, Any]:
        """Run prediction on preprocessed input"""
        try:
            X = self.preprocess(input_data)

            # Disable feature name validation (this fixes your error)
            prediction = self.model.predict(X, validate_features=False)[0]

            result = {
                "predicted_sales": round(float(prediction), 2),
                "currency": "USD",
                "note": "XGBoost prediction – original scale"
            }

            logger.info(f"Predicted sales: {prediction:.2f}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


# Singleton instance
_service_instance = None

def get_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance