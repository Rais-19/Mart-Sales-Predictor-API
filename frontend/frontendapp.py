import streamlit as st
import requests
import pandas as pd
import json
import os
#   CONFIG & STYLE


API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Mart Sales Predictor", layout="wide", page_icon="🛒")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #45a049; }
    .card { background: white; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 20px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)


#   HEADER & DESCRIPTION
st.title("🛒 Mart Sales Predictor")
st.markdown("""
Predict how much an item will sell in a specific outlet — powered by XGBoost and real sales data.

**Just fill in the product & store details** → get an instant sales forecast in USD.
""")

st.info("💡 Tip: Use realistic values (e.g. MRP between 30–300, Weight 5–20 kg) for the most accurate predictions.", icon="ℹ️")

#   SIDEBAR – CONTROLS & HELP

with st.sidebar:
    st.header("Prediction Settings")
    st.markdown("Adjust only if you want to experiment — defaults are usually best.")

    show_advanced = st.checkbox("Show optional fields", value=False)

    st.markdown("---")
    st.caption("About the model")
    st.caption("• Trained on ~8,500 historical sales records")
    st.caption("• Uses XGBoost (no log transformation)")
    st.caption("• Top features: MRP, Outlet Type, Outlet Age, Visibility")
    st.caption("• Typical error: ±$800–1200 (MAE)")

#   MAIN FORM – USER INPUT

with st.form("sales_form", clear_on_submit=False):
    st.subheader("Product Information")

    col1, col2 = st.columns(2)

    with col1:
        item_weight = st.number_input(
            "Item Weight (kg)",
            min_value=0.1, max_value=50.0, value=9.3, step=0.1,
            help="Average weight of the product"
        )

        item_visibility = st.number_input(
            "Item Visibility",
            min_value=0.0, max_value=0.35, value=0.016, step=0.001,
            help="How visible is the product on the shelf? (0 = hidden, 0.35 = very visible)"
        )

        item_mrp = st.number_input(
            "Maximum Retail Price (MRP)",
            min_value=10.0, max_value=500.0, value=249.8, step=1.0,
            help="The highest price the item can be sold for"
        )

    with col2:
        item_fat = st.selectbox(
            "Item Fat Content",
            options=["Low Fat", "Regular"],
            index=0,
            help="Fat classification of the product"
        )

        item_type = st.selectbox(
            "Item Type (optional)",
            options=["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Others", ""],
            index=0,
            help="Category of the product — leave blank if unsure"
        )

    st.subheader("Outlet / Store Information")

    col3, col4 = st.columns(2)

    with col3:
        outlet_size = st.selectbox(
            "Outlet Size",
            options=["Small", "Medium", "High"],
            index=1,
            help="Physical size of the store"
        )

        outlet_location = st.selectbox(
            "Outlet Location Tier",
            options=["Tier 1", "Tier 2", "Tier 3"],
            index=0,
            help="City/town tier (Tier 1 = most developed)"
        )

    with col4:
        outlet_type = st.selectbox(
            "Outlet Type",
            options=["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"],
            index=0,
            help="Kind of store"
        )

        outlet_year = st.number_input(
            "Outlet Establishment Year",
            min_value=1980, max_value=2025, value=1999, step=1,
            help="Year the store was opened"
        )

    # Submit button
    submitted = st.form_submit_button("Predict Sales", type="primary", use_container_width=True)


#   PREDICTION LOGIC & DISPLAY

if submitted:
    with st.spinner("Calculating predicted sales..."):
        try:
            payload = {
                "Item_Weight": item_weight,
                "Item_Fat_Content": item_fat,
                "Item_Visibility": item_visibility,
                "Item_MRP": item_mrp,
                "Outlet_Size": outlet_size,
                "Outlet_Location_Type": outlet_location,
                "Outlet_Type": outlet_type,
                "Outlet_Establishment_Year": outlet_year
            }

            if item_type and item_type != "":
                payload["Item_Type"] = item_type

            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()

                st.success("Prediction Complete!", icon="✅")

                # result card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Predicted Sales")
                st.metric(
                    label="Estimated Item Outlet Sales",
                    value=f"${data['prediction']['predicted_sales']:.2f}",
                    delta=None,
                    help="This is the forecasted total sales for this item in the selected outlet"
                )
                st.caption(data['prediction']['note'])
                st.markdown("</div>", unsafe_allow_html=True)

                # Show input summary
                with st.expander("Input values used", expanded=False):
                    st.json(payload)

            else:
                error_data = response.json()
                st.error(f"API Error ({response.status_code}): {error_data.get('detail', 'Unknown error')}")
                st.json(error_data)

        except requests.exceptions.RequestException as e:
            st.error(f"Cannot connect to API: {str(e)}")
            st.info("Make sure the FastAPI server is running (`uvicorn app:app --reload`)")

#   FOOTER

st.markdown("---")
st.caption("For educational purposes – predictions are estimates only")