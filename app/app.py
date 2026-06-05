import streamlit as st
import pandas as pd
import joblib

## Page Config ##
st.set_page_config(
    page_title="Loan Approval & Loan Amount Predictor",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
}

.big-font {
    font-size:20px !important;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

## Model Load
@st.cache_resource
def load_models():
    approval_model = joblib.load("./model/loan_approval_model.pkl")
    amount_model = joblib.load("./model/loan_amount_model.pkl")

    return approval_model, amount_model


try:
    approval_model, amount_model = load_models()
except Exception as e:
    st.error(f"Unable to load model files.\n\n{e}")
    st.stop()

## Header
st.title("🏦 Loan Approval & Loan Amount Prediction System")

st.markdown("""
This application predicts:

- Loan Approval Status
- Eligible Loan Amount
- EMI Details
""")

# =================
# SIDEBAR
# =================
st.sidebar.title("Applicant Information")

# personal details
st.sidebar.subheader("Personal Details")
no_of_dependents = st.sidebar.number_input("No. of Dependents",min_value=0,max_value=20,value=0)
education = st.sidebar.selectbox("Education",["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed",["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income (₹)",min_value=0.0,value=5000000.0)

# Loan Details
st.sidebar.subheader("Loan Details")
loan_amount_requested = st.sidebar.number_input("Requested Loan Amount (₹)",min_value=0.0,value=1000000.0,step=100000.0)
loan_term = st.sidebar.number_input("Loan Term (Months)",min_value=3,value=60)
cibil_score = st.sidebar.slider("CIBIL Score",
    300,
    900,
    700
)

# Asset Details
st.sidebar.subheader("Asset Details")
residential_asset = st.sidebar.number_input("Residential Asset Value (₹)",min_value=0.0,value=0.0)
commercial_asset = st.sidebar.number_input("Commercial Asset Value (₹)",min_value=0.0,value=0.0)
luxury_asset = st.sidebar.number_input("Luxury Asset Value (₹)",min_value=0.0,value=0.0)
bank_asset = st.sidebar.number_input("Bank Asset Value (₹)",min_value=0.0,value=0.0)

## EMI Settings
st.sidebar.subheader("EMI Settings")
interest_rate = st.sidebar.slider("Interest Rate (%)",
    5.0,
    20.0,
    10.0
)

## Prediction Button
predict_btn = st.sidebar.button(
    "🔮 Predict"
)

# =====================================================
# MAIN PREDICTION
# =====================================================

if predict_btn:

    try:
        # ENCODING
        education_encoded = (1 if education == "Graduate" else 0)
        self_employed_encoded = (1 if self_employed == "Yes" else 0)

        # FEATURE ENGINEERING
        total_assets = (residential_asset + commercial_asset + luxury_asset + bank_asset)
        loan_income_ratio = (loan_amount_requested/ max(income_annum, 1))
        asset_coverage_ratio = (total_assets/ max(loan_amount_requested, 1))
        income_per_dependent = (income_annum/ (no_of_dependents + 1))
        wealth_cibil_score = (total_assets * cibil_score)

        # CLASSIFICATION DATA
        classification_df = pd.DataFrame(
        {
            'no_of_dependents': [no_of_dependents],
            'education': [education_encoded],
            'self_employed': [self_employed_encoded],
            'income_annum': [income_annum],
            'loan_amount': [loan_amount_requested],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'total_assets': [total_assets],
            'loan_income_ratio': [loan_income_ratio],
            'asset_coverage_ratio': [asset_coverage_ratio],
            'income_per_dependent': [income_per_dependent],
            'wealth_cibil_score': [wealth_cibil_score]
        }
        )

        # APPROVAL PREDICTION

        approval_prediction = (approval_model.predict(classification_df)[0])
        approval_probability = (approval_model.predict_proba(classification_df)[0][1])
        
        st.header("Loan Approval Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Approval Probability",f"{approval_probability:.2%}")
        with col2:
            st.metric("Total Assets",f"₹ {total_assets:,.0f}")

        # REJECTED
        if approval_prediction == 0:
            st.error("❌ Loan Application Rejected")
        # APPROVED
        else:
            st.success("✅ Loan Application Approved")

            # =========================================
            # REGRESSION DATA
            # =========================================

            regression_df = pd.DataFrame({
                'no_of_dependents':[no_of_dependents],
                'education':[education_encoded],
                'self_employed':[self_employed_encoded],
                'income_annum': [income_annum],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'total_assets': [total_assets],
                'income_per_dependent': [income_per_dependent],
                'asset_coverage_ratio': [asset_coverage_ratio],
                'wealth_cibil_score': [wealth_cibil_score]
            })

            # =========================================
            # LOAN AMOUNT PREDICTION
            # =========================================

            predicted_loan_amount = (amount_model.predict(regression_df)[0])

            st.header("Eligible Loan Amount")

            st.metric("Predicted Loan Amount",f"₹ {predicted_loan_amount:,.0f}")

            # =========================================
            # EMI CALCULATION
            # =========================================

            monthly_rate = (interest_rate / 12) / 100
            n = loan_term
            emi = (
                predicted_loan_amount
                * monthly_rate
                * ((1 + monthly_rate) ** n)
            ) / (
                ((1 + monthly_rate) ** n) - 1
            )

            total_payment = emi * n
            total_interest = ( total_payment - predicted_loan_amount)

            st.header("EMI Details")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Monthly EMI",f"₹ {emi:,.0f}")
            with col2:
                st.metric("Total Payment",f"₹ {total_payment:,.0f}")
            with col3:
                st.metric("Total Interest",f"₹ {total_interest:,.0f}")

    except Exception as e:
        st.error(f"Prediction failed.\n\n{e}")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Loan Approval Prediction System | XGBoost Classification & Regression"
)
