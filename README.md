# Mart Sales Predictor – Project Report

## Goal of the Program

**Predict future sales of individual products in different retail stores** using historical sales data.

→ The program helps store managers answer the question:  
“How much of this product (e.g. milk, chips, soda) will likely be sold in this particular store next month?”

→ Real-world analogy:  
Like a weather forecast for sales — instead of saying “it will rain 70% chance”, it says “this chocolate bar in the small supermarket in Tier 2 city will probably generate ~$1,450 next period”.

## What the Program Actually Does 

1. You enter product details  
   (weight, price, visibility on shelf, fat content, type…)

2. You enter store details  
   (store size, city tier, supermarket or grocery, year opened…)

3. The program uses a trained machine learning model (XGBoost)  
   → instantly calculates an estimated sales amount in USD

4. You get a clear number + explanation  
   → example: **Predicted sales: $1,452.30 USD**

## Main Features

- Clean web interface (Streamlit) — looks nice on desktop & mobile
- Input form with descriptions and reasonable default values
- Real-time prediction via FastAPI backend
- Automatic error messages when input is wrong
- Health check & feature list endpoints (for developers)
- Model trained on ~8,500 real sales records

## Technologies Used

- Machine Learning: XGBoost Regressor
- Backend API: FastAPI
- Frontend: Streamlit
- Data handling: pandas, numpy
- Model saving: pickle

## Model Performance Summary

- Train R²: ≈ 0.79  
- Test R²: ≈ 0.56  
- Test RMSE: ≈ $1,160  
- Test MAE: ≈ $817  

(meaning on average the prediction is off by about $817 )

## How It Works – High-Level Flow

1. User fills form in browser (Streamlit)
2. Data sent to FastAPI server
3. Server loads trained XGBoost model once at startup
4. Server preprocesses input (calculates store age, encodes categories)
5. Model makes prediction
6. Result returned and shown nicely in browser

- **Limited training data**  
  Only ~8,500 rows (after cleaning) — many real retail forecasting models use 100k–1M+ records to learn subtle patterns.

- **Few strong predictive features**  
  The dataset mainly contains basic product and store attributes.  
  Missing very powerful signals such as:  
  - promotions / discounts  
  - holiday / seasonal flags  
  - competitor prices  
  - foot traffic / store location coordinates  
  - historical sales trends per item per store
**Note**  
With the available data and features, R² ≈ 0.56 is actually a realistic and respectable baseline.  
The main limiting factors are **data volume** and **feature richness**, not the algorithm or code quality.

Improving these areas (more data, richer features, log target, tuning) is the most promising path to significantly better results.


## Final Note

This project demonstrates a complete end-to-end ML application:  
data cleaning → feature engineering → model training → API serving →  user interface.

