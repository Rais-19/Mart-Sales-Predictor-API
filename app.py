from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
from schemas.prediction import (
    SalesInput,
    PredictionResponse,
    SalesPrediction,
    FeaturesResponse,
    HealthResponse
)
from services.prediction_service import get_service

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mart Sales Predictor API",
    description="Simple API to predict item outlet sales ",
    version="1.0"
)

# Get service instance 
service = get_service()


@app.get("/", tags=["General"])
def root():
    """Welcome message and API info"""
    return {
        "message": "Welcome to Mart Sales Predictor API",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "Check API and model status",
            "GET /features": "See expected input features",
            "POST /predict": "Predict sales for one item/outlet"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check endpoint"""
    try:
        # Simple check if model is loaded
        if service.model is not None:
            features_count = len(service.feature_names) if service.feature_names else "unknown"
            return {
                "status": "healthy",
                "model_loaded": True,
                "message": f"XGBoost model ready ({features_count} features)"
            }
        else:
            return {"status": "unhealthy", "model_loaded": False}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False, "error": str(e)}
        )


@app.get("/features", response_model=FeaturesResponse, tags=["Info"])
def get_features():
    """Return the list of features the model expects"""
    try:
        required = [
            "Item_Weight", "Item_Fat_Content", "Item_Visibility",
            "Item_MRP", "Outlet_Size", "Outlet_Location_Type",
            "Outlet_Type", "Outlet_Establishment_Year"
        ]
        optional = ["Item_Type", "Outlet_Identifier"]

        return {
            "required_features": required,
            "optional_features": optional,
            "model_features_count": len(service.feature_names) if service.feature_names else "unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving features: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_sales(input_data: SalesInput):
    """Predict Item_Outlet_Sales for given input"""
    try:
        logger.info(f"Prediction request received: {input_data.dict()}")
        prediction = service.predict(input_data)

        return PredictionResponse(
            input_data=input_data,
            prediction=SalesPrediction(**prediction)
        )
    except ValueError as ve:
        logger.error(f"Validation/Prediction error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)