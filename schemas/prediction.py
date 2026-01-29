from pydantic import BaseModel, Field
from typing import List, Optional


class SalesInput(BaseModel):
    """
    Input data for single sales prediction.
  
    """
    Item_Weight: float = Field(..., gt=0, description="Weight in kg")
    Item_Fat_Content: str = Field(..., pattern="^(Low Fat|Regular)$")
    Item_Visibility: float = Field(..., ge=0, le=0.35)
    Item_MRP: float = Field(..., gt=0)
    Outlet_Size: str = Field(..., pattern="^(Small|Medium|High)$")
    Outlet_Location_Type: str = Field(..., pattern="^(Tier 1|Tier 2|Tier 3)$")
    Outlet_Type: str = Field(..., pattern="^(Supermarket Type1|Supermarket Type2|Supermarket Type3|Grocery Store)$")
    Outlet_Establishment_Year: int = Field(..., ge=1980, le=2030)

    # Optional fields — can be omitted or filled with defaults
    Item_Type: Optional[str] = None
    Outlet_Identifier: Optional[str] = None


class SalesPrediction(BaseModel):
    predicted_sales: float
    currency: str = "USD"
    note: str = "Prediction from XGBoost model"


class PredictionResponse(BaseModel):
    input_data: SalesInput
    prediction: SalesPrediction


class FeaturesResponse(BaseModel):
    required_features: List[str]
    optional_features: List[str]
    model_features_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str = ""