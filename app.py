import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="PricePulse | Dynamic Pricing Engine", layout="wide")

@st.cache_resource
def load_assets():
    model = xgb.XGBRegressor()
    model.load_model("pricepulse_xgb.json")
    with open("feature_cols.json", "r") as f:
        feature_cols = list(json.load(f).values())
    df = pd.read_csv("pricepulse_data.csv")
    return model, feature_cols, df

model, feature_cols, df = load_assets()
skus = ["Wireless_Earbuds", "Smart_Watch", "Mechanical_Keyboard", "USB_C_Hub"]
default_prices = {"Wireless_Earbuds": 60.0, "Smart_Watch": 150.0, "Mechanical_Keyboard": 95.0, "USB_C_Hub": 35.0}

st.title("📈 PricePulse: Adaptive Dynamic Pricing & Revenue Engine")
st.markdown("Real-time price elasticity modeling and revenue optimization powered by **XGBoost**.")

tab1, tab2 = st.tabs(["🎯 Real-Time Price Optimizer", "📊 Market Data & Model Benchmarks"])

# --- TAB 1: Interactive Price Optimizer ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Market Conditions")
        selected_sku = st.selectbox("Select Product SKU", skus)
        base_p = default_prices[selected_sku]
        
        comp_price = st.number_input("Competitor Price ($)", value=base_p, min_value=5.0, step=1.0)
        inventory = st.slider("Current Stock Level (Units)", 10, 500, 200, 10)
        is_promo = st.checkbox("Flash Sale / Promotional Event Active")
        is_weekend = st.checkbox("Weekend Peak Traffic")
        
    with col2:
        # Simulate 60 candidate prices around competitor price
        candidate_prices = np.linspace(max(5.0, comp_price * 0.65), comp_price * 1.40, 60)
        sim_rows = []
        for p in candidate_prices:
            row = {
                "Price": p,
                "Competitor_Price": comp_price,
                "Inventory": inventory,
                "Is_Promo": int(is_promo),
                "Is_Weekend": int(is_weekend),
                "Price_Diff_Comp": p - comp_price,
            }
            for s in skus:
                row[f"SKU_{s}"] = 1 if s == selected_sku else 0
            sim_rows.append(row)
            
        sim_df = pd.DataFrame(sim_rows)[feature_cols]
        pred_demands = np.maximum(1, model.predict(sim_df))
        # Cap demand by available inventory so we don't recommend stockouts
        effective_sales = np.minimum(pred_demands, inventory)
        pred_revenues = candidate_prices * effective_sales
        
        opt_idx = int(np.argmax(pred_revenues))
        opt_price = candidate_prices[opt_idx]
        opt_demand = int(round(effective_sales[opt_idx]))
        opt_rev = pred_revenues[opt_idx]
        
        # Compare against static competitor matching
        static_idx = int(np.argmin(np.abs(candidate_prices - comp_price)))
        static_rev = pred_revenues[static_idx]
        uplift = ((opt_rev - static_rev) / max(1.0, static_rev)) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended Optimal Price", f"${opt_price:.2f}", f"{opt_price - comp_price:+.2f} vs Comp")
        m2.metric("Expected Units Sold", f"{opt_demand} units", f"Stock: {inventory}")
        m3.metric("Projected Daily Revenue", f"${opt_rev:,.2f}", f"{uplift:+.1f}% vs Static")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(candidate_prices, pred_revenues, color="#2563eb", linewidth=2.5, label="Projected Revenue ($)")
        ax.axvline(opt_price, color="#16a34a", linestyle="--", linewidth=2, label=f"Optimal Price (${opt_price:.2f})")
        ax.axvline(comp_price, color="#6b7280", linestyle=":", label=f"Competitor Price (${comp_price:.2f})")
        ax.set_title(f"Revenue Response Curve — {selected_sku}")
        ax.set_xlabel("Candidate Price ($)")
        ax.set_ylabel("Projected Revenue ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# --- TAB 2: Model Benchmarks & Historical Data ---
with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", "4.80 units", "Target: < 8.0")
    c2.metric("Test R² Score", "0.9803")
    c3.metric("SKUs Modeled", "4 Categories")
    
    st.image("revenue_curve.png", caption="Baseline Revenue Optimization Curve")
    st.subheader("Sample Pricing & Demand Logs")
    st.dataframe(df.head(50), use_container_width=True)
