import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Default Prediction", page_icon="💳", layout="wide")


model = pickle.load(open("models/full_pipeline.pkl", "rb"))

st.markdown("""
<style>
.main{ background-color: #f5f7fa; }
.stButton>button{
    width:100%; background-color: #1f77b4; color:white;
    font-size:18px; border-radius:10px; height:3em; border:none;
}
.stButton>button:hover{ background-color: #125d98; color: white; }
.title-style { text-align: center; color: #1f2937; }
.subtitle-style { text-align: center; color: #6b7280; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-style'>Loan Risk Assessment System</h1>", unsafe_allow_html=True)

# Main Form Container
with st.container():
    # Row 1: Key Profile Data
    st.subheader(" Applicant Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        education_level = st.selectbox("Education Level", ["high school", "bachelor's", "master's", "phd", "other"])
    with col2:
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)
        employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed"])
    with col3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        loan_purpose = st.selectbox("Loan Purpose", ["home", "education", "car", "business", "personal"])

    st.markdown("---")
    
    # Row 2: Specific Loan Terms
    st.subheader(" Loan Specifications")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0.0, value=15000.0)
        loan_term = st.number_input("Loan Term (Months)", min_value=1, value=36)
    with col5:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5)
        installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=500.0)
    with col6:
        grade_subgrade = st.selectbox("Risk Grade (Subgrade)", 
            ['a1','a2','a3','a4','a5','b1','b2','b3','b4','b5','c1','c2','c3','c4','c5',
             'd1','d2','d3','d4','d5','e1','e2','e3','e4','e5','f1','f2','f3','f4','f5'])
        debt_to_income_ratio = st.number_input("DTI Ratio (%)", min_value=0.0, value=20.0)

    st.markdown("---")

    # Row 3: Financial History
    st.subheader(" Credit History & Financial Depth")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0.0, value=50000.0)
        current_balance = st.number_input("Current Balance ($)", min_value=0.0, value=5000.0)
    with col8:
        num_of_open_accounts = st.number_input("Open Accounts", min_value=0, value=5)
        public_records = st.number_input("Public Records", min_value=0, value=0)
    with col9:
        num_of_delinquencies = st.number_input("Total Delinquencies", min_value=0, value=0)
        delinquency_history = st.number_input("Delinquency History (Months)", min_value=0, value=0)

# ---------------- PREDICTION LOGIC ----------------

def convert_grade(value):
    grade_map = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6}
    return grade_map[value[0]] * 10 + int(value[1])

if st.button("🔍 Run Risk Analysis"):
    # 1. Feature Engineering (must match training)
    loan_income_ratio = loan_amount / (annual_income + 1)
    grade_subgrade_numeric = convert_grade(grade_subgrade)
    total_delinquency_impact = delinquency_history + num_of_delinquencies

    # 2. Construct DataFrame (removed dropped columns)
    input_data = pd.DataFrame([{
        'age': age,
        'education_level': education_level,
        'annual_income': annual_income,
        'employment_status': employment_status,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'loan_term': loan_term,
        'installment': installment,
        'grade_subgrade': grade_subgrade_numeric,
        'num_of_open_accounts': num_of_open_accounts,
        'total_credit_limit': total_credit_limit,
        'current_balance': current_balance,
        'public_records': public_records,
        'num_of_delinquencies': num_of_delinquencies,
        'total_delinquency_impact': total_delinquency_impact,
        'loan_income_ratio': loan_income_ratio,
        'loan_purpose': loan_purpose
    }])

    # 3. Predict using Probability and the 0.94 Threshold
    # We use [:, 1] to get the probability of 'Paid Back' (1)
    prob_paid_back = model.predict_proba(input_data)[0, 1]
    
    st.markdown("### Analysis Results")
    
    # 0.94 Threshold Logic
    if prob_paid_back >= 0.80:
        st.success(f"✅ **APPROVED**: Probability of repayment is {prob_paid_back*100:.1f}%. This customer meets high-safety standards.")
    else:
        risk_score = (1 - prob_paid_back) * 100
        st.error(f"⚠️ **REJECTED**: Risk of default detected ({risk_score:.1f}% risk). This applicant does not meet the 94% safety threshold.")

    # Show confidence bar
    progress_value = float(prob_paid_back)
    progress_value = max(0.0, min(1.0, progress_value)) 

    st.progress(progress_value)