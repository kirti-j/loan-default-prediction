import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Default Prediction",page_icon="💳",layout="wide")

model=pickle.load(open(r"C:\Users\Kirti\OneDrive\Desktop\Loan Default Prediction\models\full_pipeline.pkl","rb"))

#basic css
st.markdown("""
<style>
.main{
    background-color: #f5f7fa;
}
.stButton>button{
    width:100%;
    background-color: #1f77b4;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:3em;
    border:none;
}
.stButton>button:hover{
    background-color: #125d98;
    color: white;
}
.input-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

.title-style {
    text-align: center;
    color: #1f2937;
}

.subtitle-style {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}

</style>           
        
""",unsafe_allow_html=True)

st.markdown("<h1 class='title-style'>Loan Default Prediction System</h1>",unsafe_allow_html=True)

st.markdown("""
<p class='subtitle-style'>
Predict whether a customer is likely to repay the loan using Machine Learning
</p>
""", unsafe_allow_html=True)

col1,col2=st.columns(2)

#Left column
with col1:
    st.markdown("Applicant Information")
    age=st.number_input("Age",min_value=18,max_value=100,value=30)
    gender=st.selectbox("Gender",["male","female","others"])
    marital_status=st.selectbox("Marital Status",["single","married","divorced"])
    education_level=st.selectbox("Education Level",["high school","bachelor's","master's","phd","other"])
    employment_status=st.selectbox("Employment Status",["employed","self-employed","unemployed"])
    annual_income=st.number_input("Annual Income",min_value=0.0,value=500000.0)
    monthly_income=st.number_input("Monthly Income",min_value=0.0,value=40000.0)
    credit_score=st.number_input("Credit Score",min_value=300,max_value=900,value=700)

#Right column

with col2:
    st.markdown("Loan Information")
    loan_amount=st.number_input("Loan Amount",min_value=0.0,value=200000.0)
    interest_rate=st.number_input("Interest Rate",min_value=0.0,value=10.5)
    loan_term=st.number_input("Loan Term(Months)",min_value=1,value=36)
    installment=st.number_input("Monthly Installment",min_value=0.0,value=5000.0)
    # Convert grade_subgrade
    def convert_grade(value):
        grade = value[0]
        sub = int(value[1])
        grade_map = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6}

        return grade_map[grade] * 10 + sub
    grade_subgrade = st.selectbox("Grade Subgrade",
    ['a1','a2','a3','a4','a5',
    'b1','b2','b3','b4','b5',
    'c1','c2','c3','c4','c5',
    'd1','d2','d3','d4','d5',
    'e1','e2','e3','e4','e5',
    'f1','f2','f3','f4','f5'])
        
    loan_purpose=st.selectbox("Loan Purpose", ["home", "education", "car", "business", "personal"])
    debt_to_income_ratio=st.number_input("Debt To Income Ratio",min_value=0.0,value=20.0)

st.markdown("Financial Details")
col3,col4=st.columns(2)

with col3:
    num_of_open_accounts=st.number_input("Number of Open Accounts",min_value=0,value=5)
    total_credit_limit=st.number_input("Total Credit Limit",min_value=0.0,value=500000.0)
    current_balance=st.number_input("Current Balance",min_value=0.0,value=100000.0)

with col4:
      delinquency_history = st.number_input("Delinquency History",min_value=0,value=0)
      public_records = st.number_input("Public Records",min_value=0,value=0)
      num_of_delinquencies = st.number_input("Number of Delinquencies",min_value=0,value=0)

#predict button
st.markdown("---")

if st.button("🔍 Predict Loan Status"):
    loan_income_ratio = loan_amount / (annual_income + 1)
    grade_subgrade_numeric = convert_grade(grade_subgrade)

    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education_level,
        'annual_income': annual_income,
        'monthly_income': monthly_income,
        'employment_status': employment_status,
        'loan_amount': loan_amount,
        'grade_subgrade': grade_subgrade_numeric,
        'loan_purpose': loan_purpose,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_score': credit_score,
        'interest_rate': interest_rate,
        'loan_term': loan_term,
        'installment': installment,
        'num_of_open_accounts': num_of_open_accounts,
        'total_credit_limit': total_credit_limit,
        'current_balance': current_balance,
        'delinquency_history': delinquency_history,
        'public_records': public_records,
        'num_of_delinquencies': num_of_delinquencies,
        'loan_income_ratio': loan_income_ratio
    }])

    prediction=model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ This customer is likely to repay the loan.")
    else:
        st.error("⚠️ High risk of loan default.")

