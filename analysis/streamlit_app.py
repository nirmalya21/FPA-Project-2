
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/extended_project_financials_template_v2.csv", parse_dates=["Month"])
    df.sort_values("Month", inplace=True)
    return df

df = load_data()

# Calculated fields
df["Revenue_Variance"] = df["Actual_Revenue"] - df["Planned_Revenue"]
df["Cost_Variance"] = df["Actual_Cost"] - df["Planned_Cost"]
df["Profit"] = df["Actual_Revenue"] - df["Actual_Cost"]
df["Planned_Profit"] = df["Planned_Revenue"] - df["Planned_Cost"]
df["Profit_Variance"] = df["Profit"] - df["Planned_Profit"]
df["PVM_Impact"] = (
    df["Planned_Revenue"] * (df["Price_Index"] - 1) +
    df["Planned_Revenue"] * (df["Volume_Index"] - 1) +
    df["Planned_Revenue"] * (df["Mix_Index"] - 1)
)

# Sidebar filters
st.sidebar.header("🔧 Filters")
region = st.sidebar.selectbox("Region", df["Region"].unique())
bu = st.sidebar.selectbox("Business Unit", df["Business_Unit"].unique())
project = st.sidebar.selectbox("Project", df["Project"].unique())
status = st.sidebar.selectbox("Status", df["Status"].unique())
filtered = df[(df["Region"] == region) & (df["Business_Unit"] == bu) & (df["Project"] == project) & (df["Status"] == status)]

# Title
st.title("📊 FP&A | Business Controlling | PVM | PM Dashboard")
st.caption("by Nirmalya Rajpandit")

# KPI Summary
st.subheader("📌 Key KPIs")
st.metric("Total Profit", f"€{filtered['Profit'].sum():,.0f}")
st.metric("Avg. FX Rate", f"{filtered['FX_Rate'].mean():.2f}")
st.metric("Headcount Avg", int(filtered['Headcount'].mean()))

# Revenue and Profit Trend
st.subheader(f"📈 Revenue & Profit Trend: {project}")
fig = px.line(filtered, x="Month", y=["Planned_Revenue", "Actual_Revenue", "Profit"], title="Revenue & Profit Over Time")
st.plotly_chart(fig, use_container_width=True)

# Variance Overview
st.subheader("📉 Variance Overview")
st.dataframe(filtered[["Month", "Revenue_Variance", "Cost_Variance", "Profit_Variance"]].set_index("Month"))

# PVM Analysis
st.subheader("🔍 Price-Volume-Mix (PVM) Impact")
pvm = filtered[["Month", "Price_Index", "Volume_Index", "Mix_Index", "PVM_Impact"]].set_index("Month")
st.line_chart(pvm[["PVM_Impact"]])
st.dataframe(pvm)

# CapEx and OpEx
st.subheader("🏗️ CapEx & OpEx")
st.bar_chart(filtered.set_index("Month")[["CapEx", "OpEx"]])

# Scenario Planning
st.sidebar.header("📊 Scenario Planning")
revenue_growth = st.sidebar.slider("Revenue Growth (%)", -10, 20, 5) / 100
cost_inflation = st.sidebar.slider("Cost Inflation (%)", -5, 15, 3) / 100
fx_adj = st.sidebar.slider("FX Rate Impact (%)", -10, 10, 0) / 100
headcount_adj = st.sidebar.slider("Headcount Impact (%)", -20, 20, 0) / 100

scenario_df = filtered.copy()
scenario_df["Scenario_Revenue"] = scenario_df["Actual_Revenue"] * (1 + revenue_growth)
scenario_df["Scenario_Cost"] = scenario_df["Actual_Cost"] * (1 + cost_inflation)
scenario_df["Scenario_Profit"] = scenario_df["Scenario_Revenue"] - scenario_df["Scenario_Cost"]

st.subheader("🔧 Scenario Planning – Simulated Profit")
st.line_chart(scenario_df.set_index("Month")[["Profit", "Scenario_Profit"]])

# Forecasting
st.subheader("📈 Profit Forecast (Driver-Based)")

def train_forecast_model(df, target_col):
    df = df.reset_index(drop=True)
    df["Month_Ordinal"] = df["Month"].map(lambda x: x.toordinal())
    model = LinearRegression()
    model.fit(df[["Month_Ordinal"]], df[target_col])
    future_months = pd.date_range(df["Month"].max() + pd.DateOffset(months=1), periods=3, freq="MS")
    future_ordinal = future_months.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    preds = model.predict(future_ordinal)
    return future_months, preds

months_future, profit_preds = train_forecast_model(filtered, "Profit")
months_future, pvm_preds = train_forecast_model(filtered, "PVM_Impact")

forecast_df = pd.DataFrame({"Month": months_future, "Forecast_Profit": profit_preds, "Forecast_PVM": pvm_preds})
all_combined = pd.concat([filtered[["Month", "Profit", "PVM_Impact"]], forecast_df], ignore_index=True)

fig2 = px.line(all_combined, x="Month", y=["Profit", "Forecast_Profit", "PVM_Impact", "Forecast_PVM"],
               title="Driver-Based Forecast: Profit & PVM")
st.plotly_chart(fig2, use_container_width=True)

# Export
st.subheader("📦 Export KPIs")
kpi_options = st.multiselect("Choose KPIs to export", options=["Profit", "PVM_Impact", "Revenue_Variance", "Cost_Variance", "CapEx", "OpEx"])
if kpi_options:
    st.download_button("Download CSV", data=filtered[["Month"] + kpi_options].to_csv(index=False), file_name="kpi_export.csv", mime="text/csv")
