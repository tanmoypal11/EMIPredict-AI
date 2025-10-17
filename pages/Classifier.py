import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# -----------------------
# 1️⃣ Load Model & Encoders
# -----------------------
model = joblib.load("xgb_classifier_model.joblib")
gender_le = joblib.load("gender_label_encoder.joblib")
edu_map = joblib.load("education_custom_mapping.joblib")
ohe = joblib.load("emi_scenario_ohe.joblib")

# -----------------------
# 2️⃣ Streamlit UI
# -----------------------
st.set_page_config(page_title="EMI Eligibility Predictor", page_icon="💰", layout="centered")

st.title("💳 EMI Eligibility Prediction App")
st.markdown("Fill in the details below to predict EMI eligibility:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", ["male", "female"])
    marital_status = st.selectbox("Marital Status", ["single", "married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["private", "government", "self-employed"])
    company_type = st.selectbox("Company Type", ["Small", "Mid-size", "Large Indian", "MNC", "Startup"])
    house_type = st.selectbox("House Type", ["Rented", "Family", "Own"])
    existing_loans = st.selectbox("Existing Loans", ["yes", "no"])
    emi_scenario = st.selectbox("EMI Scenario", list(ohe.categories_[0]))  # ✅ safe dropdown from trained encoder

with col2:
    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=5000, value=50000, step=1000)
    years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=2.0, step=0.1)
    monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, value=10000, step=500)
    family_size = st.number_input("Family Size", min_value=1, value=3)
    dependents = st.number_input("Dependents", min_value=0, value=1)
    school_fees = st.number_input("School Fees (₹)", min_value=0, value=0)
    college_fees = st.number_input("College Fees (₹)", min_value=0, value=0)
    travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, value=5000)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, value=10000)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0, value=5000)
    current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0, value=0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    bank_balance = st.number_input("Bank Balance (₹)", min_value=0, value=100000)
    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=20000)
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, value=500000, step=10000)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, value=12, step=1)
    max_monthly_emi = st.number_input("Max Affordable EMI (₹)", min_value=1000, value=10000, step=500)

# -----------------------
# 3️⃣ Create DataFrame
# -----------------------
user_data = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'marital_status': marital_status,
    'education': education,
    'monthly_salary': monthly_salary,
    'employment_type': employment_type,
    'years_of_employment': years_of_employment,
    'company_type': company_type,
    'house_type': house_type,
    'monthly_rent': monthly_rent,
    'family_size': family_size,
    'dependents': dependents,
    'school_fees': school_fees,
    'college_fees': college_fees,
    'travel_expenses': travel_expenses,
    'groceries_utilities': groceries_utilities,
    'other_monthly_expenses': other_monthly_expenses,
    'existing_loans': existing_loans,
    'current_emi_amount': current_emi_amount,
    'credit_score': credit_score,
    'bank_balance': bank_balance,
    'emergency_fund': emergency_fund,
    'emi_scenario': emi_scenario,
    'requested_amount': requested_amount,
    'requested_tenure': requested_tenure,
    'max_monthly_emi': max_monthly_emi
}])

# -----------------------
# 4️⃣ Preprocessing
# -----------------------
# lowercase normalization
for col in ['gender', 'marital_status', 'employment_type', 'company_type', 'house_type', 'existing_loans']:
    user_data[col] = user_data[col].str.lower()

# encode gender
user_data['gender'] = gender_le.transform(user_data['gender'])

# encode education
user_data['education'] = user_data['education'].map(edu_map)

# safely one-hot encode EMI scenario
valid_emi_options = list(ohe.categories_[0])
emi_input = user_data['emi_scenario'].iloc[0]

if emi_input not in valid_emi_options:
    st.warning(f"⚠️ '{emi_input}' not in trained EMI categories. Defaulting to zeros.")
    emi_encoded = np.zeros((1, len(ohe.get_feature_names_out(['emi_scenario']))))
else:
    emi_encoded = ohe.transform(user_data[['emi_scenario']])

emi_encoded_df = pd.DataFrame(
    emi_encoded,
    columns=ohe.get_feature_names_out(['emi_scenario']),
    index=user_data.index
)
user_data = pd.concat([user_data.drop('emi_scenario', axis=1), emi_encoded_df], axis=1)

# temporarily label encode remaining categorical features
for col in ['marital_status', 'employment_type', 'company_type', 'house_type', 'existing_loans']:
    le = LabelEncoder()
    user_data[col] = le.fit_transform(user_data[col])

# -----------------------
# 5️⃣ Feature Engineering
# -----------------------
def feature_engineering(df):
    df = df.copy()
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']
    df['expense_to_income_ratio'] = (
        df['monthly_rent'] + df['groceries_utilities'] + df['other_monthly_expenses']
    ) / df['monthly_salary']
    df['affordability_ratio'] = df['max_monthly_emi'] / df['monthly_salary']
    df['credit_utilization_ratio'] = df['requested_amount'] / (df['bank_balance'] + 1)
    df['employment_stability_score'] = np.log1p(df['years_of_employment']) * (df['credit_score'] / 100)
    df['financial_stress_index'] = (
        df['debt_to_income_ratio'] + df['expense_to_income_ratio'] + df['credit_utilization_ratio']
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

df_user_final = feature_engineering(user_data)

# -----------------------
# 6️⃣ Prediction
# -----------------------
if st.button("🔍 Predict EMI Eligibility"):
    prediction = model.predict(df_user_final)[0]
    label_map = {0: '❌ Not Eligible', 1: '⚠️ High Risk', 2: '✅ Eligible'}

    st.subheader("Prediction Result")
    st.success(f"Predicted EMI Eligibility: **{label_map[prediction]}**")
