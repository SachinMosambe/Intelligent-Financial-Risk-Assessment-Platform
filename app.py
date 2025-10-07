import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc

# ==================== CONFIG ====================
st.set_page_config(
    page_title="EMI Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
REG_MODEL_NAME = "MaxEMIRegressor"
REG_MODEL_VERSION = 2
CLASS_MODEL_NAME = "EMIEligibilityClassifier"
CLASS_MODEL_VERSION = 2
USED_LOG1P_TRANSFORM = True

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
except Exception as e:
    st.warning(f"MLflow connection issue: {e}")

# Feature lists
CLASSIFICATION_FEATURES = [
    'requested_amount', 'expense_to_income', 'requested_tenure', 'monthly_salary', 
    'groceries_utilities', 'bank_balance', 'debt_to_income', 'travel_expenses', 
    'credit_score', 'current_emi_amount', 'emergency_fund', 'other_monthly_expenses', 
    'savings_buffer_ratio', 'existing_loans', 'school_fees', 'college_fees', 
    'years_of_employment', 'employment_stability_score', 'monthly_rent'
]

REGRESSION_FEATURES = [
    'expense_to_income', 'debt_to_income', 'monthly_salary', 'current_emi_amount'
]


# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_model_by_version(model_name: str, version: int):
    """Load model from MLflow Registry (cached)"""
    try:
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_model_metadata(model_name: str, version: int):
    """Get model metadata"""
    try:
        client = mlflow.MlflowClient()
        mv = client.get_model_version(model_name, version)
        return {
            "name": model_name,
            "version": mv.version,
            "stage": mv.current_stage,
            "available": True
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ==================== FEATURE ENGINEERING ====================
def compute_engineered_features(df: pd.DataFrame, for_classification: bool = True) -> pd.DataFrame:
    """Compute engineered features matching training pipeline"""
    df = df.copy()
    salary_safe = df["monthly_salary"].replace(0, np.nan)
    
    # Core ratios
    df["debt_to_income"] = df["current_emi_amount"] / salary_safe
    total_expenses = (
        df["monthly_rent"] + df["school_fees"] + df.get("college_fees", 0) +
        df["travel_expenses"] + df["groceries_utilities"] + df["other_monthly_expenses"]
    )
    df["expense_to_income"] = total_expenses / salary_safe
    
    # Classification-only features
    if for_classification:
        df["savings_buffer_ratio"] = (df["bank_balance"] + df["emergency_fund"]) / salary_safe
        df["employment_stability_score"] = df["years_of_employment"]
    
    # Fill NaN
    ratio_cols = ["debt_to_income", "expense_to_income"]
    if for_classification:
        ratio_cols.append("savings_buffer_ratio")
    df[ratio_cols] = df[ratio_cols].fillna(0.0)
    
    return df


def prepare_model_input(df: pd.DataFrame, for_classification: bool = True) -> pd.DataFrame:
    """Prepare data with only required features"""
    df = df.copy()
    
    if "existing_loans" in df.columns:
        df["existing_loans"] = df["existing_loans"].map({"No": 0, "Yes": 1})
    
    feature_list = CLASSIFICATION_FEATURES if for_classification else REGRESSION_FEATURES
    
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0.0
    
    return df[feature_list]


# ==================== INPUT FORMS ====================
def get_common_inputs():
    """Get common inputs for both models"""
    inputs = {}
    
    st.subheader("Employment & Income")
    c1, c2 = st.columns(2)
    inputs["monthly_salary"] = c1.number_input(
        "Monthly Salary (Rs)", min_value=1000.0, max_value=10000000.0, 
        value=50000.0, step=5000.0
    )
    inputs["current_emi_amount"] = c2.number_input(
        "Current EMI (Rs)", min_value=0.0, value=0.0, step=1000.0
    )
    
    st.subheader("Monthly Expenses")
    c1, c2, c3 = st.columns(3)
    inputs["monthly_rent"] = c1.number_input("Rent (Rs)", min_value=0.0, value=10000.0, step=1000.0)
    inputs["groceries_utilities"] = c2.number_input("Groceries & Utilities (Rs)", min_value=0.0, value=8000.0, step=500.0)
    inputs["travel_expenses"] = c3.number_input("Travel (Rs)", min_value=0.0, value=3000.0, step=500.0)
    
    c4, c5, c6 = st.columns(3)
    inputs["school_fees"] = c4.number_input("School Fees (Rs)", min_value=0.0, value=0.0, step=1000.0)
    inputs["college_fees"] = c5.number_input("College Fees (Rs)", min_value=0.0, value=0.0, step=1000.0)
    inputs["other_monthly_expenses"] = c6.number_input("Other Expenses (Rs)", min_value=0.0, value=5000.0, step=500.0)
    
    return inputs


def get_classification_inputs():
    """Additional inputs for classification"""
    inputs = get_common_inputs()
    
    st.subheader("Loan Request Details")
    c1, c2 = st.columns(2)
    inputs["requested_amount"] = c1.number_input(
        "Loan Amount (Rs)", min_value=10000.0, max_value=10000000.0, 
        value=500000.0, step=10000.0
    )
    inputs["requested_tenure"] = c2.number_input(
        "Tenure (months)", min_value=6, max_value=360, value=60, step=6
    )
    
    c1, c2 = st.columns(2)
    inputs["years_of_employment"] = c1.number_input(
        "Years of Employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5
    )
    inputs["credit_score"] = c2.number_input(
        "Credit Score", min_value=300, max_value=900, value=700
    )
    
    st.subheader("Credit & Savings")
    c1, c2, c3 = st.columns(3)
    inputs["has_existing_loans"] = c1.radio("Existing Loans?", ["No", "Yes"], horizontal=True)
    inputs["bank_balance"] = c2.number_input("Bank Balance (Rs)", min_value=0.0, value=100000.0, step=10000.0)
    inputs["emergency_fund"] = c3.number_input("Emergency Fund (Rs)", min_value=0.0, value=50000.0, step=5000.0)
    
    return inputs


# ==================== MAIN APP ====================
st.title("🏦 EMI Prediction & Eligibility App")
st.markdown("**Powered by MLflow** | Get instant predictions for loan eligibility or maximum affordable EMI")

# Sidebar - Model Info
st.sidebar.header("📊 Model Information")
with st.spinner("Loading models..."):
    clf_model = load_model_by_version(CLASS_MODEL_NAME, CLASS_MODEL_VERSION)
    reg_model = load_model_by_version(REG_MODEL_NAME, REG_MODEL_VERSION)

clf_metadata = get_model_metadata(CLASS_MODEL_NAME, CLASS_MODEL_VERSION)
reg_metadata = get_model_metadata(REG_MODEL_NAME, REG_MODEL_VERSION)

if clf_metadata["available"] and clf_model:
    st.sidebar.success(f"✓ {CLASS_MODEL_NAME} v{CLASS_MODEL_VERSION}")
else:
    st.sidebar.error(f"✗ {CLASS_MODEL_NAME}: Not Available")

if reg_metadata["available"] and reg_model:
    st.sidebar.success(f"✓ {REG_MODEL_NAME} v{REG_MODEL_VERSION}")
else:
    st.sidebar.error(f"✗ {REG_MODEL_NAME}: Not Available")

st.divider()

# Mode Selection
mode = st.radio(
    "**Select Prediction Type:**",
    ("EMI Eligibility Classification", "Maximum EMI Prediction"),
    horizontal=True
)

st.divider()

# ==================== CLASSIFICATION MODE ====================
if mode == "EMI Eligibility Classification":
    st.header("📋 Loan Eligibility Prediction")
    
    if not clf_model:
        st.error(f"Classification model (v{CLASS_MODEL_VERSION}) unavailable. Check MLflow.")
        st.stop()
    
    with st.form("eligibility_form"):
        inputs = get_classification_inputs()
        submitted = st.form_submit_button("Predict Eligibility", width="stretch", type="primary")
    
    if submitted:
        with st.spinner("Processing..."):
            try:
                input_data = pd.DataFrame([inputs])
                input_data["existing_loans"] = input_data["has_existing_loans"]
                input_data = input_data.drop(columns=["has_existing_loans"], errors='ignore')
                
                input_data = compute_engineered_features(input_data, for_classification=True)
                model_input = prepare_model_input(input_data, for_classification=True)
                prediction = clf_model.predict(model_input)[0]
                
                st.divider()
                st.subheader("🎯 Prediction Result")
                
                labels = {0: "Not Eligible", 1: "High_Risk", 2: "Eligible"}
                result = labels.get(prediction, "Unknown")
                
                if prediction == 2:
                    st.success(f"**✅ {result}** for Loan")
                    st.balloons()
                elif prediction == 1:
                    st.warning(f"**⚠️ {result}** for Loan")
                else:
                    st.error(f"**❌ {result}** for Loan")
                    st.info("💡 Consider improving credit score or reducing EMIs")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Monthly Salary", f"₹{inputs['monthly_salary']:,.0f}")
                c2.metric("Requested Amount", f"₹{inputs['requested_amount']:,.0f}")
                c3.metric("Tenure", f"{inputs['requested_tenure']} months")
                c4.metric("Credit Score", f"{inputs['credit_score']}")
                
                with st.expander("📊 View Computed Features"):
                    feature_cols = ["debt_to_income", "expense_to_income", 
                                  "savings_buffer_ratio", "employment_stability_score"]
                    st.dataframe(input_data[feature_cols].T.style.format("{:.4f}"), width="stretch")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# ==================== REGRESSION MODE ====================
else:
    st.header("💰 Maximum EMI Prediction")
    
    if not reg_model:
        st.error(f"Regression model (v{REG_MODEL_VERSION}) unavailable. Check MLflow.")
        st.stop()
    
    with st.form("regression_form"):
        inputs = get_common_inputs()
        submitted = st.form_submit_button("Predict Max EMI", width="stretch", type="primary")
    
    if submitted:
        with st.spinner("Calculating..."):
            try:
                input_data = pd.DataFrame([inputs])
                input_data = compute_engineered_features(input_data, for_classification=False)
                model_input = prepare_model_input(input_data, for_classification=False)
                
                prediction_log = reg_model.predict(model_input)[0]
                prediction = max(0, np.expm1(prediction_log))
                
                st.divider()
                st.subheader("🎯 Prediction Result")
                st.success(f"### Maximum Affordable EMI: ₹{prediction:,.2f}")
                
                available = max(0, prediction - inputs['current_emi_amount'])
                utilization = (inputs['current_emi_amount'] / prediction * 100) if prediction > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Monthly Salary", f"₹{inputs['monthly_salary']:,.2f}")
                c2.metric("Current EMI", f"₹{inputs['current_emi_amount']:,.2f}")
                c3.metric("Available Capacity", f"₹{available:,.2f}")
                
                st.subheader("📊 EMI Utilization")
                st.progress(float(min(utilization / 100, 1.0)))
                st.caption(f"Using {utilization:.1f}% of EMI capacity")
                
                if available > 0:
                    st.success(f"✅ You can take additional EMI: ₹{available:,.2f}/month")
                else:
                    st.warning("⚠️ You're at maximum EMI capacity")
                
                with st.expander("📊 View Computed Features"):
                    st.dataframe(input_data[["debt_to_income", "expense_to_income"]].T.style.format("{:.4f}"), width="stretch")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>Powered by MLflow & Streamlit | Models from MLflow Registry</small><br>
    <small>⚠️ Prediction tool only. Actual approval depends on lender policies.</small>
</div>
""", unsafe_allow_html=True)