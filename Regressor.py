import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Saved Models & Encoders
# -----------------------------
xgb_reg = joblib.load("xgb_regressor_model.joblib")
gender_le = joblib.load("gender_label_encoder.joblib")
edu_map = joblib.load("education_custom_mapping.joblib")
ohe = joblib.load("emi_scenario_ohe.joblib")

# -----------------------------
# Helper: Feature Engineering
# -----------------------------
def feature_engineering(df):
    df = df.copy()
    df["debt_to_income_ratio"] = df["current_emi_amount"] / df["monthly_salary"]
    df["expense_to_income_ratio"] = (
        df["monthly_rent"] + df["groceries_utilities"] + df["other_monthly_expenses"]
    ) / df["monthly_salary"]
    df["credit_utilization_ratio"] = df["requested_amount"] / (df["bank_balance"] + 1)
    df["employment_stability_score"] = np.log1p(df["years_of_employment"]) * (
        df["credit_score"] / 100
    )
    df["financial_stress_index"] = (
        df["debt_to_income_ratio"]
        + df["expense_to_income_ratio"]
        + df["credit_utilization_ratio"]
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏦 EMI Prediction App")
st.write("Estimate your **maximum affordable EMI** based on your financial profile.")

# --- Dropdowns and Inputs ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    company_type = st.selectbox("Company Type", ["Small", "Startup", "Mid-size", "Large Indian", "MNC"])
    years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 2.0, step=0.1)
    credit_score = st.number_input("Credit Score", 300, 900, 700)
    emi_scenario = st.selectbox(
        "EMI Scenario",
        ["Vehicle EMI", "Home Appliances EMI", "Education EMI", "Personal Loan EMI", "E-commerce Shopping EMI"]
    )

with col2:
    monthly_salary = st.number_input("Monthly Salary (₹)", 5000, 2000000, 50000)
    house_type = st.selectbox("House Type", ["Rented", "Family", "Own"])
    monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000)
    family_size = st.number_input("Family Size", 1, 10, 3)
    dependents = st.number_input("Dependents", 0, 10, 2)
    travel_expenses = st.number_input("Travel Expenses (₹)", 0, 50000, 5000)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 50000, 10000)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 50000, 5000)
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 200000, 10000)
    bank_balance = st.number_input("Bank Balance (₹)", 0, 5000000, 100000)
    emergency_fund = st.number_input("Emergency Fund (₹)", 0, 1000000, 20000)
    requested_amount = st.number_input("Requested Loan Amount (₹)", 0, 5000000, 500000)
    requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 24)
    emi_eligibility = st.selectbox("EMI Eligibility", ["Not_Eligible", "High_Risk", "Eligible"])

# -----------------------------
# Prepare Input
# -----------------------------
if st.button("🔍 Predict EMI"):
    df_input = pd.DataFrame(
        {
            "age": [age],
            "gender": [gender],
            "marital_status": [marital_status],
            "education": [education],
            "monthly_salary": [monthly_salary],
            "employment_type": [employment_type],
            "years_of_employment": [years_of_employment],
            "company_type": [company_type],
            "house_type": [house_type],
            "monthly_rent": [monthly_rent],
            "family_size": [family_size],
            "dependents": [dependents],
            "school_fees": [0],
            "college_fees": [0],
            "travel_expenses": [travel_expenses],
            "groceries_utilities": [groceries_utilities],
            "other_monthly_expenses": [other_monthly_expenses],
            "existing_loans": [existing_loans],
            "current_emi_amount": [current_emi_amount],
            "credit_score": [credit_score],
            "bank_balance": [bank_balance],
            "emergency_fund": [emergency_fund],
            "emi_scenario": [emi_scenario],
            "requested_amount": [requested_amount],
            "requested_tenure": [requested_tenure],
            "emi_eligibility": [emi_eligibility],
        }
    )

    # lowercase categorical
    for col in [
        "gender",
        "marital_status",
        "employment_type",
        "company_type",
        "house_type",
        "existing_loans",
        "emi_eligibility",
    ]:
        df_input[col] = df_input[col].astype(str).str.lower()

    # encode gender
    df_input["gender"] = gender_le.transform(df_input["gender"])

    # encode education
    df_input["education"] = df_input["education"].map(edu_map)

    # label encode remaining categorical columns
    from sklearn.preprocessing import LabelEncoder
    for col in ["marital_status", "employment_type", "company_type", "house_type", "existing_loans", "emi_eligibility"]:
        le = LabelEncoder()
        df_input[col] = le.fit_transform(df_input[col])

    # one-hot encode emi_scenario
    emi_encoded = ohe.transform(df_input[["emi_scenario"]])
    emi_encoded_df = pd.DataFrame(
        emi_encoded, columns=ohe.get_feature_names_out(["emi_scenario"])
    )
    df_input = pd.concat([df_input.drop("emi_scenario", axis=1), emi_encoded_df], axis=1)

    # feature engineering
    df_final = feature_engineering(df_input)

    # align features
    expected_features = xgb_reg.get_booster().feature_names
    for col in expected_features:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[expected_features]

    # predict
    predicted_emi = xgb_reg.predict(df_final)[0]
    st.success(f"💰 Predicted Maximum Affordable EMI: ₹{predicted_emi:,.2f}")
